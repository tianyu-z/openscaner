const credentialInput = document.querySelector("#credential");
const saveCredentialButton = document.querySelector("#save-credential");
const uploadForm = document.querySelector("#upload-form");
const filesInput = document.querySelector("#files");
const jobList = document.querySelector("#job-list");
const preview = document.querySelector("#result-preview");
const filterButtons = [...document.querySelectorAll(".filter")];

let activeFilter = "active";
let selectedJobId = null;
let objectUrls = [];

credentialInput.value = localStorage.getItem("openscaner_credential") || "";

saveCredentialButton.addEventListener("click", () => {
  localStorage.setItem("openscaner_credential", credentialInput.value);
  refreshJobs();
});

credentialInput.addEventListener("change", () => {
  localStorage.setItem("openscaner_credential", credentialInput.value);
});

filterButtons.forEach((button) => {
  button.addEventListener("click", () => {
    activeFilter = button.dataset.filter;
    filterButtons.forEach((item) => item.classList.toggle("is-active", item === button));
    refreshJobs();
  });
});

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const data = new FormData();
  [...filesInput.files].forEach((file) => data.append("images", file));
  const response = await fetch("/api/v1/jobs", {
    method: "POST",
    headers: credentialHeaders(),
    body: data,
  });
  if (!response.ok) {
    await renderNotice(`Upload failed (${response.status})`);
    return;
  }
  const job = await response.json();
  selectedJobId = job.id;
  filesInput.value = "";
  await refreshJobs();
  await loadJob(job.id);
});

function credentialHeaders() {
  const credential = localStorage.getItem("openscaner_credential") || credentialInput.value;
  if (!credential) return {};
  return {
    "X-API-Key": credential,
    "X-OpenScaner-Password": credential,
  };
}

function html(value) {
  return String(value ?? "").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "\"": "&quot;",
    "'": "&#39;",
  })[character]);
}

function filteredJobs(jobs) {
  const done = new Set(["succeeded", "partially_failed"]);
  const active = new Set(["queued", "running"]);
  if (activeFilter === "active") return jobs.filter((job) => active.has(job.status));
  if (activeFilter === "done") return jobs.filter((job) => done.has(job.status));
  if (activeFilter === "failed") return jobs.filter((job) => job.status === "failed");
  if (activeFilter === "expired") return jobs.filter((job) => job.status === "expired");
  return jobs;
}

async function refreshJobs() {
  const response = await fetch("/api/v1/jobs", {headers: credentialHeaders()});
  if (!response.ok) {
    jobList.innerHTML = `<div class="notice">Unable to load jobs (${response.status})</div>`;
    return;
  }
  const jobs = filteredJobs(await response.json());
  if (jobs.length === 0) {
    jobList.innerHTML = '<div class="notice">No jobs</div>';
    return;
  }
  jobList.innerHTML = jobs.map((job) => `
    <button class="job-row ${job.id === selectedJobId ? "is-selected" : ""}" type="button" data-job-id="${html(job.id)}">
      <strong>${html(job.id)}</strong>
      <span class="job-meta">
        <span class="status ${html(job.status)}">${html(job.status)}</span>
        <span>${job.done_count}/${job.total_count}</span>
        <span>${job.failed_count} failed</span>
      </span>
    </button>
  `).join("");
  document.querySelectorAll(".job-row").forEach((row) => {
    row.addEventListener("click", () => loadJob(row.dataset.jobId));
  });
}

async function loadJob(jobId) {
  selectedJobId = jobId;
  const response = await fetch(`/api/v1/jobs/${encodeURIComponent(jobId)}`, {headers: credentialHeaders()});
  if (!response.ok) {
    await renderNotice(`Job unavailable (${response.status})`);
    return;
  }
  const job = await response.json();
  await renderJob(job);
  await refreshJobs();
}

async function renderJob(job) {
  const succeeded = job.items.find((item) => item.status === "succeeded" && item.result);
  if (!succeeded) {
    clearObjectUrls();
    preview.innerHTML = `
      <div class="detail-header">
        <div>
          <h2>${html(job.id)}</h2>
          <div class="job-meta"><span class="status ${html(job.status)}">${html(job.status)}</span></div>
        </div>
      </div>
      ${renderItems(job.items)}
      <pre>${html(JSON.stringify(job, null, 2))}</pre>
    `;
    return;
  }

  const overlayUrl = await authenticatedObjectUrl(succeeded.result.overlay_url);
  const rectifiedUrl = await authenticatedObjectUrl(succeeded.result.rectified_url);
  preview.innerHTML = `
    <div class="detail-header">
      <div>
        <h2>${html(job.id)}</h2>
        <div class="job-meta">
          <span class="status ${html(job.status)}">${html(job.status)}</span>
          <span>${job.done_count}/${job.total_count}</span>
          <span>${job.failed_count} failed</span>
        </div>
      </div>
      <div class="download-actions">
        ${job.download_url ? '<button id="download-zip" type="button">Download ZIP</button>' : ""}
      </div>
    </div>
    <div class="preview-grid">
      <section class="preview-panel">
        <h3>Overlay</h3>
        ${overlayUrl ? `<img src="${overlayUrl}" alt="Overlay preview">` : '<div class="notice">Overlay unavailable</div>'}
      </section>
      <section class="preview-panel">
        <h3>Rectified</h3>
        ${rectifiedUrl ? `<img src="${rectifiedUrl}" alt="Rectified output">` : '<div class="notice">Rectified unavailable</div>'}
      </section>
    </div>
    <div class="metrics">
      <div class="metric"><span>Status</span><strong>${html(succeeded.result.status)}</strong></div>
      <div class="metric"><span>Confidence</span><strong>${html(succeeded.result.confidence)}</strong></div>
      <div class="metric"><span>Adapter</span><strong>${html(succeeded.result.adapter)}</strong></div>
      <div class="metric"><span>Elapsed</span><strong>${html(succeeded.result.elapsed_ms)} ms</strong></div>
    </div>
    ${renderItems(job.items)}
    <pre>${html(JSON.stringify(succeeded.result, null, 2))}</pre>
  `;
  const downloadZip = document.querySelector("#download-zip");
  if (downloadZip) {
    downloadZip.addEventListener("click", () => downloadBlob(job.download_url, `${job.id}-results.zip`));
  }
}

function renderItems(items) {
  return `
    <div class="item-list">
      ${items.map((item) => `
        <div class="item-row">
          <strong>${html(item.original_filename)}</strong>
          <span class="item-meta">
            <span class="status ${html(item.status)}">${html(item.status)}</span>
            <span>${html(item.id)}</span>
          </span>
        </div>
      `).join("")}
    </div>
  `;
}

async function authenticatedObjectUrl(url) {
  if (!url) return null;
  const response = await fetch(url, {headers: credentialHeaders()});
  if (!response.ok) return null;
  const objectUrl = URL.createObjectURL(await response.blob());
  objectUrls.push(objectUrl);
  return objectUrl;
}

async function downloadBlob(url, filename) {
  const response = await fetch(url, {headers: credentialHeaders()});
  if (!response.ok) {
    await renderNotice(`Download failed (${response.status})`);
    return;
  }
  const objectUrl = URL.createObjectURL(await response.blob());
  const anchor = document.createElement("a");
  anchor.href = objectUrl;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(objectUrl);
}

async function renderNotice(message) {
  clearObjectUrls();
  preview.innerHTML = `<div class="notice">${html(message)}</div>`;
}

function clearObjectUrls() {
  objectUrls.forEach((url) => URL.revokeObjectURL(url));
  objectUrls = [];
}

refreshJobs();
setInterval(refreshJobs, 3000);
