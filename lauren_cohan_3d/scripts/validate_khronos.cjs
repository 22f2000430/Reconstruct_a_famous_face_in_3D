#!/usr/bin/env node
"use strict";

const fs = require("node:fs");
const path = require("node:path");

const root = path.resolve(__dirname, "..");
const validatorPath = path.join(
  root,
  ".validator",
  "node_modules",
  "gltf-validator",
);
const input = path.resolve(process.argv[2] || path.join(root, "output/lauren_cohan.glb"));
const reportPath = path.resolve(
  process.argv[3] || path.join(root, "reports/khronos_validation.json"),
);

if (!fs.existsSync(validatorPath)) {
  throw new Error(
    "Khronos validator package is missing. Install it with: " +
      "npm install --prefix .validator --no-save gltf-validator@2.0.0-dev.3.10",
  );
}
if (!fs.existsSync(input)) {
  throw new Error(`GLB not found: ${input}`);
}

const validator = require(validatorPath);
const bytes = new Uint8Array(fs.readFileSync(input));
validator
  .validateBytes(bytes, { uri: path.basename(input), maxIssues: 1000 })
  .then((report) => {
    const result = {
      validator: "Khronos glTF Validator",
      version: require(path.join(validatorPath, "package.json")).version,
      file: input,
      valid: report.issues.numErrors === 0,
      numErrors: report.issues.numErrors,
      numWarnings: report.issues.numWarnings,
      numInfos: report.issues.numInfos,
      numHints: report.issues.numHints,
      issues: report.issues.messages,
    };
    fs.mkdirSync(path.dirname(reportPath), { recursive: true });
    fs.writeFileSync(reportPath, `${JSON.stringify(result, null, 2)}\n`);
    console.log(JSON.stringify(result, null, 2));
    if (!result.valid) {
      process.exitCode = 1;
    }
  })
  .catch((error) => {
    console.error(`Khronos glTF Validator failed: ${error.message}`);
    process.exitCode = 1;
  });
