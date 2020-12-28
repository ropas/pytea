const path = require('path');

module.exports = (_, { mode }) => {
    return {
        mode: process.env.NODE_ENV || 'development',
        entry: './src/client/index.tsx',
        devtool: 'inline-source-map',
        module: {
            rules: [
                {
                    test: /\.tsx?$/,
                    loader: 'ts-loader',
                    exclude: /node_modules/,
                    options: {
                        configFile: 'tsconfig.client.json',
                    },
                },
                {
                    test: /\.scss$/,
                    use: ['style-loader', 'css-loader', 'sass-loader'],
                },
            ],
        },
        resolve: {
            extensions: ['.tsx', '.ts', '.js', '.css', '.scss'],
        },
        output: {
            filename: 'app.js',
            path: path.resolve(__dirname, 'public/js'),
        },
    };
};
