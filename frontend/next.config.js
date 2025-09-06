/** @type {import('next').NextConfig} */
const nextConfig = {
  // Only enable static export for production builds
  ...(process.env.BUILD_MODE === 'export' && { output: 'export' }),
  webpack: (config) => {
    config.module.rules.push({
      test: /\.glsl$/,
      use: ['raw-loader'],
    });
    return config;
  },
  async redirects() {
    return [
      {
        source: '/v1',
        destination: '/',
        permanent: true,
      },
      {
        source: '/v1/:path*',
        destination: '/:path*',
        permanent: true,
      },
    ];
  },
};

// Set assetPrefix only in production/export mode
if (process.env.NODE_ENV === 'production' && process.env.BUILD_MODE === 'export') {
  nextConfig.assetPrefix = '/static';
}

module.exports = nextConfig;