# Express TypeScript Application

A TypeScript Express application scaffolded with express-generator-typescript.

## Features

- **Language**: TypeScript
- **Framework**: Express.js
- **Database**: none
- **Authentication**: disabled
- **WebSocket**: none
- **View Engine**: none

## Project Structure

```
bin/              # Startup scripts
controllers/      # Request handlers
models/           # Data models
public/           # Static assets
routes/           # Route definitions
services/         # Business logic
sockets/          # WebSocket handlers
views/            # View templates
utils/            # Utility functions
middleware/       # Express middleware
config/           # Configuration files
```

## Getting Started

### Prerequisites

- Node.js (v14 or higher)
- npm or yarn
  

### Installation

1. Clone the repository

```
git clone --recursive <repository-url>
cd <project-directory>
```
1.1 

If you've already cloned without `--recursive`:
```
git submodule update --init --recursive
```

1.2
Setup speaker embeddings:
```
mkdir -p FreeVC/speaker_embeddings
# Add instructions for obtaining speaker embeddings
```
1.3
Download pretrained models:
```
mkdir -p FreeVC/pretrained_models
# Add download instructions for the pretrained model
```

2. Install dependencies

```
npm install
```

3. Environment Variables

Create a `.env` file in the root directory with the following variables:

```
PORT=3000
NODE_ENV=development
CLIENT_URL=http://localhost:3000
```

### Development

Start the development server:

```
npm run dev
```

### Build

Build for production:

```
npm run build
```

### Production

Start the production server:

```
npm start
```

## API Documentation

The API documentation is available at `/api-docs` when the server is running.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
