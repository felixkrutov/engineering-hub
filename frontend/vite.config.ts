import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  // Указываем Vite, что все пути должны начинаться с /hub/
  base: '/hub/',
  server: {
    host: '0.0.0.0',
    port: 3000,
  }
})
