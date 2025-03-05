import { Router } from "express";
import { ExampleRoutes } from "./example";

export function initializeRoutes(): Router {
    const router = Router();

    router.get("/", (req, res) => {
        res.json({
            message: "Welcome to Express TypeScript API"
        });
    });

    ExampleRoutes.create(router);
    return router;
}

export default initializeRoutes;