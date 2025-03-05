//!
"#!/usr/bin/env ts-node";

import { Server } from "../src/server";

/**
 * Normalize a port into a number, string, or false.
 */
function normalizePort(val: string): number {
    const port = parseInt(val, 10);

    if (isNaN(port)) {
        return 3000;
    }

    if (port >= 0) {
        return port;
    }

    return 3000;
}

/**
 * Get port from environment and store in Express.
 */
const port = normalizePort(process.env.PORT || "3000");

/**
 * Create HTTP server.
 */
const server = Server.bootstrap();

/**
 * Listen on provided port, on all network interfaces.
 */
server.listen(port);