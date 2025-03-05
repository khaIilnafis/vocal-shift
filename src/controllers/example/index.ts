import {
  getAllController,
  getByIdController,
  createController,
  updateController,
  deleteController,
} from "./exampleController";

export class ExampleController {
  constructor() {
    this.getAll = getAllController();
    this.getById = getByIdController();
    this.create = createController();
    this.update = updateController();
    this.delete = deleteController();
  }

  getAll: ReturnType<typeof getAllController>;
  getById: ReturnType<typeof getByIdController>;
  create: ReturnType<typeof createController>;
  update: ReturnType<typeof updateController>;
  delete: ReturnType<typeof deleteController>;
}

export default ExampleController;
