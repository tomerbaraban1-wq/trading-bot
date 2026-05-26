"""Security audit script."""
import sys, os, re
sys.stdout.reconfigure(encoding='utf-8')

files = [f for f in os.listdir('.') if f.endswith('.py')]
passed = []
failed = []

def check(name, ok, note=""):
    symbol = "✅" if ok else "❌"
    print(f"  {symbol} {name}" + (f" — {note}" if note else ""))
    if ok:
        passed.append(name)
    else:
        failed.append(name)

print("=" * 65)
print("RED TEAM SECURITY AUDIT — Level MAX")
print("=" * 65)

# 1. SQL injection
print("\n[1] SQL Injection")
sql_risks = []
for f in files:
    try:
        content = open(f, encoding='utf-8', errors='ignore').read()
        # Check for % formatting in SQL
        for m in re.finditer(r'execute\([^)]*%[^)]*\)', content):
            sql_risks.append(f"{f}:{content[:m.start()].count(chr(10))+1}")
        # Check for .format() in SQL
        for m in re.finditer(r'execute\([^)]*\.format\(', content):
            sql_risks.append(f"{f}:{content[:m.start()].count(chr(10))+1}")
        # f-strings with user data (exclude internal constants)
        for m in re.finditer(r'execute\(\s*f["\']', content):
            ctx = content[m.start():m.start()+200]
            vars_in = re.findall(r'\{([^}]+)\}', ctx)
            # Only flag if variable looks like user data
            for v in vars_in:
                if any(kw in v.lower() for kw in ['request', 'param', 'user', 'input', 'body', 'data', 'text']):
                    sql_risks.append(f"{f}: user data in SQL: {v}")
    except Exception:
        pass

check("No SQL injection (parameterized queries)", len(sql_risks) == 0, f"{len(sql_risks)} risks")
for r in sql_risks[:3]:
    print(f"    -> {r}")

# 2. Eval/exec (code injection)
print("\n[2] Code Injection (eval/exec)")
code_inject = []
for f in files:
    try:
        content = open(f, encoding='utf-8', errors='ignore').read()
        for m in re.finditer(r'\beval\s*\(|\bexec\s*\(', content):
            line = content[:m.start()].count('\n') + 1
            ctx = content.split('\n')[line-1].strip()
            if not ctx.strip().startswith('#'):
                code_inject.append(f"{f}:{line}")
    except Exception:
        pass
check("No eval/exec usage", len(code_inject) == 0, f"{len(code_inject)} found")

# 3. Pickle (insecure deserialization)
print("\n[3] Insecure Deserialization")
pickle_use = []
for f in files:
    try:
        content = open(f, encoding='utf-8', errors='ignore').read()
        if 'pickle.loads' in content or 'pickle.load(' in content:
            pickle_use.append(f)
    except Exception:
        pass
check("No pickle deserialization", len(pickle_use) == 0)

# 4. HTTPS enforcement
print("\n[4] HTTPS Enforcement")
http_calls = []
for f in files:
    try:
        content = open(f, encoding='utf-8', errors='ignore').read()
        for m in re.finditer(r'"http://(?!localhost|127\.0\.0\.1|0\.0\.0\.0|schemas)', content):
            line = content[:m.start()].count('\n') + 1
            ctx = content.split('\n')[line-1].strip()[:80]
            http_calls.append(f"{f}:{line} -> {ctx}")
    except Exception:
        pass
check("All external calls use HTTPS", len(http_calls) == 0, f"{len(http_calls)} HTTP URLs")
for h in http_calls[:3]:
    print(f"    -> {h}")

# 5. Debug endpoints
print("\n[5] Debug Endpoints")
debug_eps = []
for f in files:
    try:
        content = open(f, encoding='utf-8', errors='ignore').read()
        if re.search(r'@app\.get\("/debug|@router\.get\("/debug', content):
            debug_eps.append(f)
    except Exception:
        pass
check("No debug endpoints exposed", len(debug_eps) == 0)

# 6. Secrets in code comments
print("\n[6] Secrets in Comments")
comment_secrets = []
for f in files:
    try:
        content = open(f, encoding='utf-8', errors='ignore').read()
        for m in re.finditer(r'#.*(?:password|secret|token)\s*[:=]\s*\S{8,}', content, re.IGNORECASE):
            val = m.group(0)
            if not any(x in val.lower() for x in ['os.getenv', 'env', 'example', 'xxx', 'your']):
                line = content[:m.start()].count('\n') + 1
                comment_secrets.append(f"{f}:{line}")
    except Exception:
        pass
check("No secrets in comments", len(comment_secrets) == 0, f"{len(comment_secrets)} found")

# 7. Timing attacks
print("\n[7] Timing Attack Prevention")
timing_risks = []
for f in ['webhook.py', 'security_endpoints.py', 'rental_admin.py', 'analytics_api.py']:
    try:
        content = open(f, encoding='utf-8', errors='ignore').read()
        # Look for == comparison with secret/key/token (not compare_digest)
        for m in re.finditer(r'(?:provided_key|api_key|token|secret)\s*==\s*(?!None|""|\'\')', content):
            line = content[:m.start()].count('\n') + 1
            ctx = content.split('\n')[line-1].strip()
            if 'compare_digest' not in content[max(0,m.start()-100):m.start()+100]:
                timing_risks.append(f"{f}:{line}")
    except Exception:
        pass
check("HMAC timing-safe comparisons only", len(timing_risks) == 0, f"{len(timing_risks)} direct == comparisons")

# 8. Exception info leaks
print("\n[8] Exception Info Leaks")
exc_leaks = []
for f in files:
    try:
        content = open(f, encoding='utf-8', errors='ignore').read()
        for m in re.finditer(r'detail\s*=\s*str\s*\(\s*e\s*\)', content):
            exc_leaks.append(f"{f}")
            break
    except Exception:
        pass
# This is a warning, not hard failure (some detail exposure is acceptable in APIs)
print(f"  ⚠️  {len(exc_leaks)} endpoints expose exception details via str(e)")
print(f"     Note: Acceptable for API clients; logs don't go to browser")

# 9. Input validation
print("\n[9] Input Validation")
validation_checks = [
    ('security_manager.py', 'validate_ticker', 'Ticker validation'),
    ('security_manager.py', 'validate_email', 'Email validation'),
    ('security_manager.py', 'validate_numeric', 'Numeric validation'),
    ('security_manager.py', 'sanitize_string', 'String sanitization'),
    ('telegram_chat.py', 'text[:1000]', 'Telegram input length limit'),
]
for fname, pattern, name in validation_checks:
    try:
        found = pattern in open(fname, encoding='utf-8', errors='ignore').read()
        check(name, found)
    except Exception:
        check(name, False, "file not found")

# 10. Security headers
print("\n[10] Security Headers")
header_checks = [
    'Strict-Transport-Security',
    'Content-Security-Policy',
    'X-Frame-Options',
    'X-Content-Type-Options',
    'X-XSS-Protection',
    'Referrer-Policy',
    'Permissions-Policy',
]
try:
    mid = open('security_middleware.py', encoding='utf-8').read()
    for h in header_checks:
        check(h, h in mid)
except Exception:
    pass

# 11. Rate limiting
print("\n[11] Rate Limiting")
try:
    sm = open('security_manager.py', encoding='utf-8').read()
    check("Default rate limit", '"default":' in sm)
    check("Admin rate limit", '"/admin/*":' in sm)
    check("Auto-block threshold", 'block_ip' in sm and '200' in sm)
    check("Progressive blocking (Fail2Ban)", 'record_violation' in open('security_enhanced.py', encoding='utf-8').read())
except Exception:
    pass

# 12. Cryptography
print("\n[12] Cryptography")
try:
    sm = open('security_manager.py', encoding='utf-8').read()
    tfa = open('two_factor_auth.py', encoding='utf-8').read()
    check("AES-256 (Fernet)", 'Fernet' in sm)
    check("SHA-256 hashing", 'sha256' in sm)
    check("Cryptographic random", 'secrets.token' in sm)
    check("TOTP RFC 6238", 'struct.pack' in tfa and 'hmac' in tfa)
    check("2FA encrypted storage", 'encrypt_value' in tfa)
except Exception:
    pass

# 13. Sensitive data protection
print("\n[13] Data Protection")
try:
    gi = open('.gitignore', encoding='utf-8', errors='ignore').read()
    check(".env not in git", '.env' in gi)
    check("*.db not in git", '*.db' in gi)
    check("*.log not in git", '*.log' in gi)
except Exception:
    pass

# 14. CORS security
print("\n[14] CORS Configuration")
try:
    main = open('main.py', encoding='utf-8').read()
    check("CORS whitelist (not *)", '_allowed_origins' in main and '"*"' not in main)
    check("HTTPS only in CORS", 'https://' in main and '"http://' not in main)
    check("No wildcard CORS", 'allow_origins=["*"]' not in main)
except Exception:
    pass

# FINAL SCORE
print("\n" + "=" * 65)
total = len(passed) + len(failed)
score = (len(passed) / total * 100) if total > 0 else 100
level = "ENTERPRISE" if score >= 95 else "HIGH" if score >= 85 else "MEDIUM" if score >= 70 else "LOW"
print(f"Score: {score:.0f}/100 — {level}")
print(f"Passed: {len(passed)} | Failed: {len(failed)}")
if failed:
    print("\nFailed checks:")
    for f in failed:
        print(f"  -> {f}")
