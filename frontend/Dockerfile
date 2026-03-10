# =============================================================================
# Legal-CoPilot Frontend — Placeholder
# Full Next.js implementation is a separate task.
# =============================================================================

FROM node:20-alpine

WORKDIR /app

# Minimal health indicator
RUN echo '{"name":"vakil-frontend","version":"0.1.0"}' > package.json

EXPOSE 3000

CMD ["node", "-e", "require('http').createServer((req,res)=>{ res.writeHead(200,{'Content-Type':'text/html'}); res.end('<h1>Legal-CoPilot Frontend</h1><p>Next.js implementation coming soon.</p><p>Backend API: <a href=\"http://localhost:8000/docs\">http://localhost:8000/docs</a></p>'); }).listen(3000, ()=>console.log('Frontend placeholder on :3000'))"]