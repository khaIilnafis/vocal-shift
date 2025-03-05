import { Router } from "express";
import { ExampleController } from "../controllers/example";

export class ExampleRoutes {
    static create(router: Router) {
        const controller = new ExampleController();
        router.get("/examples", controller.getAll);
        router.get("/examples/:id", controller.getById);
        router.post("/examples", controller.create);
        router.put("/examples/:id", controller.update);
        router.delete("/examples/:id", controller.delete);
    }
}

export default ExampleRoutes;