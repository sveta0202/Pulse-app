import { defineRailway, github, project, service } from "railway/iac";

// Replace before running `railway config plan`. Use a dedicated Railway project.
const repository = "YOUR_GITHUB_OWNER/YOUR_GITHUB_REPO";

export default defineRailway(() => {
  const pulse = service("pulse", {
    source: github(repository, { branch: "main" }),
    healthcheck: "/healthz",
    healthcheckTimeout: 120,
    env: { MODEL_THREADS: "1" },
  });
  return project("pulse", { resources: [pulse] });
});
