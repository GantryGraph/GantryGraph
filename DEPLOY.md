# Deployment Guide — gantrygraph.com

## Stack

```
GitHub (source) → GitHub Actions (build) → Vercel (host) ← Cloudflare (DNS + CDN)
```

---

## 1. Vercel — first deploy

```bash
# Install Vercel CLI
npm i -g vercel

# Link the project (run from the website/ directory)
cd website
vercel link

# First deploy (creates the project in Vercel dashboard)
vercel --prod
```

In the Vercel dashboard → Project Settings:
- **Root Directory**: `website`
- **Build Command**: `pip install -r requirements.txt && python -m mkdocs build`
- **Output Directory**: `site`
- **Install Command**: *(leave empty)*
- **Framework Preset**: Other

---

## 2. Custom domain in Vercel

1. Vercel dashboard → Project → **Domains**
2. Add `gantrygraph.com`
3. Add `www.gantrygraph.com`
4. Vercel gives you two records to add in Cloudflare:

| Type  | Name | Value                     |
|-------|------|---------------------------|
| A     | `@`  | `76.76.21.21`             |
| CNAME | `www`| `cname.vercel-dns.com`    |

---

## 3. Cloudflare DNS

1. Log in to **dash.cloudflare.com** → select `gantrygraph.com`
2. Go to **DNS** → **Records**
3. Add the two records from step 2

> **Important**: Set the proxy status to **DNS only** (grey cloud ☁️) for both records.
> Vercel handles SSL — leaving Cloudflare proxy on causes certificate conflicts.

4. Cloudflare → **SSL/TLS** → set mode to **Full**

---

## 4. GitHub Actions secrets

Go to **GitHub repo → Settings → Secrets and variables → Actions** and add:

| Secret | Where to find it |
|--------|-----------------|
| `VERCEL_TOKEN` | Vercel dashboard → Settings → Tokens → Create |
| `VERCEL_ORG_ID` | `.vercel/project.json` after `vercel link` (field `orgId`) |
| `VERCEL_PROJECT_ID` | `.vercel/project.json` after `vercel link` (field `projectId`) |

After this, every push to `main` that touches `website/` or `src/gantrygraph/`
triggers a production deploy automatically.

---

## 5. Local preview

```bash
cd website
pip install -r requirements.txt
pip install -e "../[dev]"   # so mkdocstrings can read the source
python -m mkdocs serve      # → http://127.0.0.1:8000
```

---

## 6. Verify everything works

```bash
# Check DNS propagation
dig gantrygraph.com +short
# Should return: 76.76.21.21

# Check SSL
curl -I https://gantrygraph.com
# Should return: HTTP/2 200
```

---

## Architecture diagram

```
Push to main
     │
     ▼
GitHub Actions (.github/workflows/deploy-docs.yml)
     │
     ├─ pip install gantrygraph + mkdocs deps
     ├─ mkdocs build --strict  (output: website/site/)
     │
     ▼
Vercel (serves website/site/ as static files)
     │
     ▼
Cloudflare DNS  (A record → 76.76.21.21)
     │
     ▼
gantrygraph.com
```
