/* eslint-disable @typescript-eslint/no-var-requires */
const path = require('path');
const CopyPlugin = require('copy-webpack-plugin');
const { TsconfigPathsPlugin } = require('tsconfig-paths-webpack-plugin');
const { CleanWebpackPlugin } = require('clean-webpack-plugin');
const { monorepoResourceNameMapper } = require('../../build/lib/webpack');
const nodeExternals = require('webpack-node-externals');

const outPath = path.resolve(__dirname, 'dist');
const typeshedFallback = path.resolve(__dirname, '..', 'pyright-internal', 'typeshed-fallback');
const pylibImplements = path.resolve(__dirname, '..', 'pytea', 'pylib');

module.exports = (_, { mode }) => {
    return {
        context: __dirname,
        cache: true,
        target: 'node',
        devtool: mode === 'development' ? 'source-map' : 'nosources-source-map',
        stats: {
            all: false,
            errors: true,
            warnings: true,
        },
        mode: process.env.NODE_ENV || 'development',
        entry: './src/server/server.ts',
        module: {
            rules: [
                {
                    test: /\.tsx?$/,
                    loader: 'ts-loader',
                    exclude: /node_modules/,
                    options: {
                        configFile: 'tsconfig.server.json',
                        experimentalWatchApi: true,
                    },
                },
            ],
        },
        resolve: {
            extensions: ['.tsx', '.ts', '.js'],
            plugins: [new TsconfigPathsPlugin({ extensions: ['.ts', '.js'], configFile: './tsconfig.server.json' })],
        },
        output: {
            filename: '[name].js',
            path: outPath,
            devtoolModuleFilenameTemplate:
                mode === 'development' ? '../[resource-path]' : monorepoResourceNameMapper('pytea'),
        },
        plugins: [
            new CleanWebpackPlugin(),
            new CopyPlugin({
                patterns: [
                    { from: typeshedFallback, to: 'typeshed-fallback' },
                    { from: pylibImplements, to: 'pylib' },
                ],
            }),
        ],
        optimization: {
            removeAvailableModules: false,
            removeEmptyChunks: false,
            splitChunks: {
                cacheGroups: {
                    defaultVendors: {
                        name: 'vendor',
                        test: /[\\/]node_modules[\\/]/,
                        chunks: 'all',
                        priority: -10,
                    },
                    pyright: {
                        name: 'pyright-internal',
                        chunks: 'all',
                        test: /[\\/]pyright-internal[\\/]/,
                        priority: -20,
                    },
                    pytea: {
                        name: 'pytea',
                        chunks: 'all',
                        test: /[\\/]pytea[\\/]/,
                        priority: -30,
                    },
                },
            },
        },
        externals: [nodeExternals()],
    };
};
