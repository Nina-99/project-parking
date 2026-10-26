FROM node:22-alpine AS builder

WORKDIR /app
COPY package.json ./
RUN npm install

COPY . .
RUN npm run build

FROM nginx:alpine

LABEL org.opencontainers.image.title="Parking Intelligence UI"
LABEL org.opencontainers.image.description="Operations panel"
LABEL org.opencontainers.image.authors="Nina"
LABEL org.opencontainers.image.licenses="Proprietary"

COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
