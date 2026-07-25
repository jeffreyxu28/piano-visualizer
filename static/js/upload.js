// Uploads a file to /api/transcribe and reports upload progress so the
// loading screen can show real feedback instead of a fake progress bar.

export function transcribeFile(file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/transcribe");

    xhr.upload.addEventListener("progress", (event) => {
      if (event.lengthComputable) {
        const pct = event.loaded / event.total;
        onProgress?.({ phase: "uploading", pct });
      }
    });

    xhr.upload.addEventListener("load", () => {
      onProgress?.({ phase: "processing", pct: 1 });
    });

    xhr.addEventListener("load", () => {
      let body;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        reject(new Error("The server returned an unexpected response."));
        return;
      }

      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(body);
      } else {
        reject(new Error(body?.error || body?.detail || `Request failed (${xhr.status}).`));
      }
    });

    xhr.addEventListener("error", () => {
      reject(new Error("Network error while uploading the file."));
    });

    xhr.addEventListener("abort", () => {
      reject(new Error("Upload was cancelled."));
    });

    const formData = new FormData();
    formData.append("file", file);
    xhr.send(formData);
  });
}
