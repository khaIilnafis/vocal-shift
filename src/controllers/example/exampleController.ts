import { Request, Response, NextFunction } from "express";
import { exec } from "child_process";
import fetch from "node-fetch";
import formidable from "formidable";
import fs from "fs";
import path from "path";
/**
 * Example model type for demonstration
 */
interface Example {
  id: number;
  name: string;
  description: string;
  isActive: boolean;
}
const UPLOAD_FOLDER = path.join(__dirname, "../../../uploads");
const execPromise = (command: string, id: string): Promise<string> =>
  new Promise((resolve, reject) => {
    exec(command, (error, stdout, stderr) => {
      if (error) return reject({ error, id });
      resolve(stdout ? stdout : `${id}:${stderr}`);
    });
  });
/**
 * Factory function that creates a controller to get all examples
 */
export function getAllController() {
  return async function getAll(
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> {
    try {
      const examples = [
        {
          id: 1,
          name: "Example 1",
          description: "Description 1",
          isActive: true,
        },
        {
          id: 2,
          name: "Example 2",
          description: "Description 2",
          isActive: true,
        },
      ];

      res.json(examples);
    } catch (error) {
      console.error("Error fetching examples:", error);
      next(error);
    }
  };
}

/**
 * Factory function that creates a controller to get example by ID
 */
export function getByIdController() {
  return async function getById(
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> {
    try {
      const id = parseInt(req.params.id, 10);

      const example = {
        id: id,
        name: "Example " + id,
        description: "Description " + id,
        isActive: true,
      };

      res.json(example);
    } catch (error) {
      console.error("Error fetching example:", error);
      next(error);
    }
  };
}

/**
 * Factory function that creates a controller to create a new example
 */
export function createController() {
  return async function create(
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> {
    const form = formidable({
      multiples: false,
      uploadDir: UPLOAD_FOLDER,
      keepExtensions: true,
      filename: (name, ext, part) => {
        return `${Date.now()}-${part.originalFilename}`;
      },
    });

    form.parse(req, async (err, fields, files) => {
      if (err) {
        console.log(err);
        return res.status(500).json({ error: "File upload failed" });
      }
      console.log(files);
      const videoFile = files.video;
      if (!videoFile) {
        return res.status(400).json({ error: "No video file uploaded" });
      }

      const videoPath = videoFile[0].filepath; // Correct way to access file path in Formidable v3+
      const audioFilename = `${Date.now()}_audio.wav`;
      const audioPath = path.join(UPLOAD_FOLDER, audioFilename);
      const outputVideoPath = `./uploads/${Date.now()}_output.mp4`;

      try {
        // Extract audio from video using ffmpeg
        await execPromise(
          `ffmpeg -i ${videoPath} -q:a 0 -map a ${audioPath}`,
          "EXTRACT"
        );

        // Send the extracted audio to the Python script for inference
        const response = await fetch("http://127.0.0.1:8000/inference", {
          method: "POST",
          body: JSON.stringify({
            audio_path: audioPath,
            audio_filename: audioFilename,
            voice_type: "kid",
          }),
          headers: { "Content-Type": "application/json" },
        });
        if (response.status !== 200) {
          const err = await response.json();
          console.log(err);
          res.status(400).json(err);
          return;
        }
        const { output_audio } = (await response.json()) as any;
        console.log(output_audio);
        // Merge processed audio back into the video
        console.log(
          `Merging: Video path: ${videoPath}, Audio path: ${output_audio}`
        );

        // Verify both files exist before merging
        if (!fs.existsSync(videoPath)) {
          throw new Error(`Video file does not exist: ${videoPath}`);
        }
        if (!fs.existsSync(output_audio)) {
          throw new Error(`Audio file does not exist: ${output_audio}`);
        }

        await execPromise(
          `ffmpeg -y -i "${videoPath}" -i "${output_audio}" -map 0:v -map 1:a -c:v copy -c:a aac -strict experimental "${outputVideoPath}"`,
          "MERGE"
        );

        res.json({
          message: "Voice conversion successful!",
          video: outputVideoPath,
        });
      } catch (error) {
        console.error("Error processing request:", error);
        res.status(500).json({ error: "Voice processing failed" });
      }
    });
  };
}

/**
 * Factory function that creates a controller to update an example
 */
export function updateController() {
  return async function update(
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> {
    try {
      const id = parseInt(req.params.id, 10);

      const {
        name: name,
        description: description,
        isActive: isActive,
      } = req.body;

      const updatedExample = {
        id: id,
        name: name,
        description: description,
        isActive: isActive,
      };

      res.json(updatedExample);
    } catch (error) {
      console.error("Error updating example:", error);
      next(error);
    }
  };
}

/**
 * Factory function that creates a controller to delete an example
 */
export function deleteController() {
  return async function deleteExample(
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> {
    try {
      const id = parseInt(req.params.id, 10);

      res.json({
        message: `Example ${id} deleted successfully`,
      });
    } catch (error) {
      console.error("Error deleting example:", error);
      next(error);
    }
  };
}
