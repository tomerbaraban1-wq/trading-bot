const localtunnel = require('localtunnel');
const fs = require('fs');
const path = require('path');

const port = parseInt(process.argv[2] || '8000');
const urlFile = path.join(__dirname, 'tunnel_url.txt');

(async () => {
    try {
        const tunnel = await localtunnel({ port });
        console.log(`TUNNEL_URL=${tunnel.url}`);
        fs.writeFileSync(urlFile, tunnel.url, 'utf-8');
        tunnel.on('close', () => process.exit(0));
    } catch (e) {
        console.error('Tunnel error:', e.message);
        process.exit(1);
    }
})();
