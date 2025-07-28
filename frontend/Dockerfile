# Stage 1: Build the React application
FROM node:18-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

# Stage 2: Serve the static files with the standard Nginx
FROM nginx:1.25-alpine
COPY --from=build /app/dist /usr/share/nginx/html
