# 🛡️ Security Documentation

מערכת אבטחה מקיפה לבוט המסחר - הגנה ברמת ייצור.

---

## 🎯 שכבות אבטחה (Defense in Depth)

```
שכבה 1: Network/Transport
   ├── HTTPS only (Render enforced)
   ├── TLS 1.2+
   └── DDoS protection (Cloudflare/Render)

שכבה 2: Application
   ├── Rate limiting (per-IP, per-endpoint)
   ├── Brute force protection
   ├── Injection detection (SQL, XSS, path traversal)
   ├── Security headers (CSP, HSTS, etc.)
   └── Input validation

שכבה 3: Authentication
   ├── API key authentication
   ├── HMAC timing-safe comparison
   ├── 2FA (TOTP + backup codes)
   └── Webhook signature verification

שכבה 4: Authorization
   ├── Admin-only endpoints
   ├── Tier-based permissions (rental)
   └── Resource-level checks

שכבה 5: Data
   ├── Encryption at rest (AES-256 via Fernet)
   ├── Encrypted API keys in DB
   ├── Audit logging
   └── No plaintext secrets in logs
```

---

## 🔐 הגדרת אבטחה - .env

### **משתני סביבה חיוניים:**

```bash
# Critical - בלי זה הבוט פתוח לחלוטין!
ADMIN_API_KEY=               # ניצור עם: python -c 'import secrets; print(secrets.token_urlsafe(32))'
WEBHOOK_SECRET=              # ניצור באותה דרך
ENCRYPTION_KEY=              # ניצור עם: python -c 'import secrets, base64; print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode())'

# Important
LICENSE_SECRET_KEY=          # למערכת השכרה
ANALYTICS_API_KEY=           # להגנת API endpoints
TELEGRAM_BOT_TOKEN=          # Bot token
TELEGRAM_CHAT_ID=            # Chat ID

# Optional but recommended
QUIET_HOURS_START=23
QUIET_HOURS_END=7
```

### **יצירת מפתחות מאובטחים:**

```bash
# Generate ADMIN_API_KEY (32 chars)
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Generate ENCRYPTION_KEY (32 bytes, base64)
python -c "import secrets, base64; print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode())"

# Generate WEBHOOK_SECRET
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

---

## 🛡️ מערכות אבטחה פעילות

### 1. **Advanced Rate Limiter**

```python
מגבלות ברירת מחדל:
- Default: 100 req/min
- /admin/*: 20 req/min
- /api/v1/*: 60 req/min
- /telegram/webhook: 30 req/min
- /webhook/*: 30 req/min
- Auth endpoints: 5 req/min

הגנה אוטומטית:
- 200+ req/min = IP blocked לשעה
- Sliding window (1 דקה)
- Per-IP + per-endpoint + per-user
```

### 2. **Brute Force Protection**

```python
- 5 כשלי auth → block לשעה
- חלון 15 דקות
- Per-user + per-IP tracking
- Auto-alert למנהל
```

### 3. **Injection Detection**

```python
מזהה ומונע:
✅ SQL Injection (union select, drop table, ' OR 1=1)
✅ XSS (<script>, javascript:, onerror=)
✅ Path Traversal (../, /etc/passwd)
✅ NoSQL Injection
✅ Command Injection (חלקי)

פעולות:
- Block immediately
- Log critical event
- Alert admin
```

### 4. **Security Headers**

כל תגובה כוללת:
```
Strict-Transport-Security: max-age=31536000
Content-Security-Policy: default-src 'self'; ...
X-Frame-Options: DENY
X-Content-Type-Options: nosniff
X-XSS-Protection: 1; mode=block
Referrer-Policy: strict-origin-when-cross-origin
Permissions-Policy: geolocation=(), microphone=(), camera=()
```

### 5. **Encryption at Rest**

- Fernet (AES-128-CBC + HMAC-SHA256)
- Sensitive fields encrypted in DB:
  - 2FA secrets
  - Backup codes
  - API credentials (when stored)
  - License keys

### 6. **Audit Logging**

כל event רגיש נשמר:
- Login attempts (success/fail)
- Rate limit hits
- Brute force detection
- Admin actions
- Data exports
- Config changes
- Suspicious requests

### 7. **Two-Factor Authentication (2FA)**

```
✅ TOTP (RFC 6238 compliant)
✅ Google Authenticator / Authy
✅ Backup codes (10 one-time use)
✅ Time-window tolerance
✅ Replay attack prevention
```

---

## 🚨 Endpoints אבטחה

### **Admin Endpoints** (requires `X-Admin-Key`)

```bash
GET  /security/status          # סטטוס אבטחה כללי
GET  /security/dashboard       # דאשבורד HTML
GET  /security/audit-log       # יומן אבטחה
GET  /security/blocked-ips     # IPs חסומים
POST /security/block-ip        # חסום IP ידנית
POST /security/unblock-ip      # שחרר IP
GET  /security/config-check    # בדיקת קונפיגורציה
POST /security/2fa/setup       # הגדרת 2FA
```

### **Public Auth Endpoints**

```bash
POST /security/2fa/verify      # אימות 2FA code
```

---

## 📊 בדיקת מצב האבטחה

### **דרך API:**
```bash
curl -H "X-Admin-Key: YOUR_KEY" \
  https://your-bot.onrender.com/security/status
```

### **דרך Dashboard:**
```
https://your-bot.onrender.com/security/dashboard
```

---

## 🔍 בדיקת אבטחה (Self-Audit)

### **בדיקה ראשונית:**

1. **משתני סביבה:**
```bash
curl -H "X-Admin-Key: $ADMIN_API_KEY" \
  https://your-bot.onrender.com/security/config-check
```

2. **יומן אירועים:**
```bash
curl -H "X-Admin-Key: $ADMIN_API_KEY" \
  "https://your-bot.onrender.com/security/audit-log?severity=critical"
```

3. **IPs חסומים:**
```bash
curl -H "X-Admin-Key: $ADMIN_API_KEY" \
  https://your-bot.onrender.com/security/blocked-ips
```

---

## 🛠️ הגדרת 2FA למנהל

### שלב 1: יצירת secret
```bash
curl -X POST \
  -H "X-Admin-Key: YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "admin", "email": "you@example.com"}' \
  https://your-bot.onrender.com/security/2fa/setup
```

### שלב 2: סריקת QR code
Response יכיל `qr_image_url` - סרוק עם Google Authenticator

### שלב 3: שמירת backup codes
Response יכיל 10 backup codes - שמור במקום מאובטח

### שלב 4: אימות לכל פעולה רגישה
```bash
curl -X POST \
  -d '{"user_id": "admin", "code": "123456"}' \
  https://your-bot.onrender.com/security/2fa/verify
```

---

## ⚠️ Best Practices

### **DO ✅**

✅ הגדר את כל משתני סביבה (ADMIN_KEY, ENCRYPTION_KEY וכו')  
✅ השתמש ב-2FA למנהל  
✅ בדוק audit log לעיתים תכופות  
✅ סובב מפתחות (rotate) כל 90 ימים  
✅ השתמש ב-HTTPS בלבד  
✅ שמור backup codes במקום מאובטח  
✅ הגדר Quiet Hours לשינה רגועה  
✅ בדוק /security/dashboard ביומיומי

### **DON'T ❌**

❌ אל תשתמש במפתחות ברירת מחדל  
❌ אל תשלח מפתחות בlogs  
❌ אל תפעיל מצב debug ב-production  
❌ אל תפתח את הAPI endpoints ללא authentication  
❌ אל תאחסן secrets בקוד  
❌ אל תשתמש ב-`*` ב-CORS  
❌ אל תתעלם מאירועים critical  

---

## 🚨 תגובה לאירועי אבטחה

### **כשמתגלה אירוע critical:**

1. **בדוק את האירוע:**
```bash
curl -H "X-Admin-Key: $KEY" \
  "https://bot/security/audit-log?severity=critical&days=1"
```

2. **חסום את ה-IP:**
```bash
curl -X POST -H "X-Admin-Key: $KEY" \
  -d '{"ip": "1.2.3.4", "duration_seconds": 86400}' \
  https://bot/security/block-ip
```

3. **סובב מפתחות:**
   - Generate חדש: `python -c 'import secrets; print(secrets.token_urlsafe(32))'`
   - Update ב-Render: Settings → Environment
   - Restart: Service → Manual Deploy

4. **בדוק שאין דליפת מידע:**
   - Check trade_log
   - Check rental_subscribers
   - Check security_audit_log

---

## 📋 OWASP Top 10 - Coverage

| OWASP Risk | סטטוס | הגנה |
|------------|--------|------|
| **A01 - Broken Access Control** | ✅ | Admin keys + 2FA + tier-based |
| **A02 - Cryptographic Failures** | ✅ | Fernet (AES-256) + TLS + HMAC |
| **A03 - Injection** | ✅ | Pattern detection + parameterized queries |
| **A04 - Insecure Design** | ✅ | Defense in depth + threat modeling |
| **A05 - Security Misconfiguration** | ✅ | Security headers + config checks |
| **A06 - Vulnerable Components** | ⏳ | תלוי בעדכון dependencies |
| **A07 - Auth Failures** | ✅ | Brute force protection + 2FA |
| **A08 - Software Integrity** | ✅ | HMAC signatures + webhook validation |
| **A09 - Logging Failures** | ✅ | Comprehensive audit log |
| **A10 - SSRF** | ✅ | Outbound URL validation |

---

## 🔐 Compliance Considerations

### **GDPR (אם רלוונטי):**
- ✅ Encryption at rest
- ✅ Audit logging
- ✅ Data minimization
- ⏳ Right to deletion (manual via DB)
- ⏳ Data export endpoint needed

### **SOC 2 (אם רלוונטי):**
- ✅ Access controls
- ✅ Audit logging
- ✅ Change management (git)
- ✅ Incident response procedures
- ⏳ Penetration testing recommended

### **PCI DSS:**
- ❌ Not applicable (לא מטפלים בכרטיסי אשראי ישירות)
- Use Stripe/PayPal for payments

---

## 🎯 Security Roadmap

### **Phase 1 - DONE ✅**
- [x] Rate limiting
- [x] Brute force protection
- [x] Injection detection
- [x] Security headers
- [x] Encryption at rest
- [x] Audit logging
- [x] 2FA infrastructure
- [x] Security dashboard

### **Phase 2 - To Implement**
- [ ] WAF integration (Cloudflare)
- [ ] Penetration testing
- [ ] Automated dependency scanning
- [ ] Secret rotation automation
- [ ] Anomaly detection (ML-based)
- [ ] Honeypot endpoints

### **Phase 3 - Advanced**
- [ ] HSM for key storage
- [ ] Zero-trust architecture
- [ ] Bug bounty program
- [ ] SOC 2 certification

---

## 📞 Security Contacts

- **Internal:** Check audit log + security dashboard
- **External vulnerabilities:** Report to admin email

---

**עדכון אחרון:** 2026-05-26 | **גרסה:** 1.0 | **סטטוס:** 🟢 ACTIVE
