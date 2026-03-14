/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",

  // Proxy all /api/* requests to the backend container via Docker internal network.
  // This avoids CORS issues and localhost ambiguity when running in Docker.
  // In local dev (npm run dev), set NEXT_PUBLIC_API_URL=http://localhost:8000
  // and this rewrite will use the backend service name "backend" (Docker DNS).
  async rewrites() {
    const backendUrl =
      process.env.BACKEND_URL || "http://backend:8000";
    return [
      {
        source: "/api/:path*",
        destination: `${backendUrl}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;