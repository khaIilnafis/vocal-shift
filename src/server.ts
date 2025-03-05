import dotenv from "dotenv";

dotenv.config({
    path: ".env"
});

import express, { Application, Request, Response, NextFunction } from "express";
import helmet from "helmet";
import cors from "cors";
import morgan from "morgan";
import path from "path";
import http from "http";
// Database imports;
// Auth imports;
import { initializeRoutes } from "./routes";

/**
* The server.
*
* @class Server
*/
export class Server {
    public app: Application;
    public server: http.Server;
    public port: number | string;

    /**
    * Bootstrap the application.
    *
    * @class Server
    * @method bootstrap
    * @static
    * @return {ng.auto.IInjectorService} Returns the newly created injector for this app.
    */
    static bootstrap() {
        return new Server();
    }

    /**
    * Constructor.
    *
    * @class Server
    * @constructor
    */
    constructor() {
        this.app = express();
        this.server = http.createServer(this.app);
        this.port = process.env.PORT || 3000;
        this.initializeMiddlewares();
        this.initializeWebSockets();
        this.initializeRoutes();
        this.initializeErrorHandling();
    }

    /**
    * Initialize application middlewares.
    *
    * @class Server
    * @method initializeMiddlewares
    * @private
    */
    initializeMiddlewares() {
        this.app.use(helmet());
        this.app.use(cors());
        this.app.use(morgan("dev"));
        this.app.use(express.json());

        this.app.use(express.urlencoded({
            extended: true
        }));

        this.app.set("views", path.join(__dirname, "views"));
        this.app.set("view engine", "");
    }

    /**
    * Initialize WebSocket server.
    *
    * @class Server
    * @method initializeWebSockets
    * @private
    */
    initializeWebSockets() {
        console.log("WebSockets not configured");
    }

    /*
    * Initialize API routes.
    *
    * @class Server
    * @method initializeRoutes
    * @private
    */
    initializeRoutes() {
        const router = initializeRoutes();
        this.app.use("/api", router);

        this.app.use("/api/*", (req: Request, res: Response) => {
            res.status(404).json({
                error: "Not Found"
            });
        });

        this.app.get("/", (req: Request, res: Response) => {
            res.render("index", {
                title: "Express TypeScript App"
            });
        });
    }

    /**
    * Initialize error handlers.
    *
    * @class Server
    * @method initializeErrorHandling
    * @private
    */
    initializeErrorHandling() {
        this.app.use((req: Request, res: Response) => {
            res.status(404).json({
                message: "Not Found"
            });
        });

        this.app.use((error: any, req: Request, res: Response, next: NextFunction) => {
            console.error(error);

            res.status(500).json({
                message: "Internal Server Error"
            });
        });
    }

    /**
    * Start the server.
    *
    * @class Server
    * @method listen
    * @param {number} port The port to listen on
    * @public
    */
    listen(port: number): void {
        this.server.listen(port, () => {
            console.log(`Server running on port ${port}`);
        });

        this.server.on("error", err => {
            const bind = typeof port === "string" ? "Pipe " + port : "Port " + port;

            switch (err.name) {
            case "EACCES":
                console.error(bind + err.message);
                process.exit(1);
                break;
            case "EADDRINUSE":
                console.error(bind + err.message);
                process.exit(1);
                break;
            default:
                throw err;
            }
        });
    }
}