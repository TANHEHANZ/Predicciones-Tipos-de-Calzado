import { uploadFile } from "../service/PostImage";

document.addEventListener("DOMContentLoaded", () => {
  const fileInput = document.querySelector("#file-input");
  const submitButton = document.querySelector("#submit-button");
  const resultContainer = document.querySelector(".link-card-grid");

  submitButton?.addEventListener("click", async () => {
    const file = fileInput?.files?.[0];
    if (file) {
      try {
        const response = await uploadFile(
          "http://192.168.0.5:5000",
          "/predict",
          file
        );
        const result = await response.json();
        if (resultContainer) {
          resultContainer.innerHTML = result.predictions
            .map(
              (prediction) => `
              <li>
                <Card
                  prov="${prediction.confidence}"
                  Probabilidad="${prediction.confidence.toFixed(2)}"
                  body="${prediction.label}"
                />
              </li>
            `
            )
            .join("");
        }
      } catch (error) {
        console.error("Error uploading file:", error);
      }
    }
  });
});
