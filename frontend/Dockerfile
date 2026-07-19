# Production frontend image: build the Vite bundle, then serve it with nginx.
# Build context is this directory (frontend/); used by BOTH docker-compose and
# Railway (`railway up ./frontend --path-as-root -s ui`), so there is exactly one
# frontend image definition.

# --- build stage ---
FROM node:20-alpine AS build
WORKDIR /app

# Install deps against the lockfile for reproducible builds.
COPY package.json package-lock.json* ./
RUN npm ci

# Build the static bundle (tsc type-check + vite build -> /app/dist).
COPY . .
RUN npm run build

# --- serve stage ---
FROM nginx:1.27-alpine

# Static assets.
COPY --from=build /app/dist /usr/share/nginx/html

# nginx config template; the base image renders templates with envsubst at start,
# substituting ${BACKEND_URL} and ${PORT} into the rendered config.
COPY nginx/default.conf.template /etc/nginx/templates/default.conf.template

# Defaults for docker-compose; Railway overrides these (its injected PORT plus
# BACKEND_URL/BACKEND_HOST pointing at the deployed api service).
ENV BACKEND_URL=http://api:8000
ENV BACKEND_HOST=api
ENV PORT=80

EXPOSE 80
# Default nginx CMD/entrypoint from the base image handles template rendering.
