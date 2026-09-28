/**
 * app.js - E-Commerce Sentiment Analysis Web Application
 * Handles unified review ingestion, instant file processing, debug live model training,
 * and batch summary analytics.
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements - Unified Input Box
  const unifiedForm = document.getElementById("unified-form");
  const reviewInput = document.getElementById("review-input");
  const analyzeBtn = document.getElementById("analyze-btn");
  const browseBtn = document.getElementById("browse-btn");
  const fileInput = document.getElementById("file-input");
  const loadSampleBtn = document.getElementById("load-sample-btn");
  const clearBtn = document.getElementById("clear-btn");
  const unifiedCard = document.getElementById("unified-input-box");
  const sampleChips = document.querySelectorAll(".squircle-chip");
  const fileLoadingIndicator = document.getElementById("file-loading-indicator");

  // DOM Elements - Single Review Result
  const singleResultContainer = document.getElementById("single-result-container");
  const resultSentimentBadge = document.getElementById("result-sentiment-badge");
  const resultStars = document.getElementById("result-stars");
  const resultStarsVal = document.getElementById("result-stars-val");
  const polarityScoreVal = document.getElementById("polarity-score-val");
  const polarityFill = document.getElementById("polarity-fill");
  const resultConfidence = document.getElementById("result-confidence");
  const probPositive = document.getElementById("prob-positive");
  const probNeutral = document.getElementById("prob-neutral");
  const probNegative = document.getElementById("prob-negative");
  const keyCuesSection = document.getElementById("key-cues-section");
  const keyCuesList = document.getElementById("key-cues-list");

  // DOM Elements - Debug Mode & Live Training
  const debugToggleCheckbox = document.getElementById("debug-toggle-checkbox");
  const debugFeedbackPanel = document.getElementById("debug-feedback-panel");
  const feedbackCounterBadge = document.getElementById("feedback-counter-badge");
  const starPickBtns = document.querySelectorAll(".star-pick-btn");
  const starPickerVal = document.getElementById("star-picker-val");
  const pillPickBtns = document.querySelectorAll(".pill-pick");
  const teachModelBtn = document.getElementById("teach-model-btn");
  const debugFeedbackMsg = document.getElementById("debug-feedback-msg");

  // DOM Elements - Batch Results
  const batchSection = document.getElementById("batch-results-section");
  const batchFilenameDisplay = document.getElementById("batch-filename-display");
  const metricTotalCount = document.getElementById("metric-total-count");
  const metricOverallSentiment = document.getElementById("metric-overall-sentiment");
  const metricAvgPolarity = document.getElementById("metric-avg-polarity");
  const metricPosCount = document.getElementById("metric-pos-count");
  const metricPosPct = document.getElementById("metric-pos-pct");
  const metricNeuCount = document.getElementById("metric-neu-count");
  const metricNeuPct = document.getElementById("metric-neu-pct");
  const metricNegCount = document.getElementById("metric-neg-count");
  const metricNegPct = document.getElementById("metric-neg-pct");
  const distBarPos = document.getElementById("dist-bar-pos");
  const distBarNeu = document.getElementById("dist-bar-neu");
  const distBarNeg = document.getElementById("dist-bar-neg");
  const filterCountAll = document.getElementById("filter-count-all");
  const filterCountPos = document.getElementById("filter-count-pos");
  const filterCountNeu = document.getElementById("filter-count-neu");
  const filterCountNeg = document.getElementById("filter-count-neg");
  const filterPills = document.querySelectorAll(".filter-btn-group .squircle-pill");
  const tableSearch = document.getElementById("table-search");
  const reviewsTbody = document.getElementById("reviews-tbody");
  const exportCsvBtn = document.getElementById("export-csv-btn");

  // Application State
  let currentSingleReviewText = "";
  let currentBatchData = null;
  let activeFilter = "all";
  let debugModeActive = localStorage.getItem("nlp_debug_mode") === "true";
  let selectedFeedbackRating = 5;
  let selectedFeedbackSentiment = "Positive";

  // ==========================================================================
  // Helper Functions
  // ==========================================================================

  function getBadgeClass(sentiment) {
    if (sentiment === "Positive") return "sentiment-positive";
    if (sentiment === "Negative") return "sentiment-negative";
    return "sentiment-neutral";
  }

  function renderStars(rating) {
    const full = Math.round(rating);
    let str = "";
    for (let i = 1; i <= 5; i++) {
      str += i <= full ? "★ " : "☆ ";
    }
    return str.trim();
  }

  function formatPolarity(polarity) {
    const prefix = polarity > 0 ? "+" : "";
    return `${prefix}${polarity.toFixed(2)}`;
  }

  // ==========================================================================
  // Debug Mode & Live Training Initialization
  // ==========================================================================

  debugToggleCheckbox.checked = debugModeActive;

  debugToggleCheckbox.addEventListener("change", (e) => {
    debugModeActive = e.target.checked;
    localStorage.setItem("nlp_debug_mode", debugModeActive);
    syncDebugPanelVisibility();
  });

  function syncDebugPanelVisibility() {
    if (debugModeActive && !singleResultContainer.classList.contains("hidden")) {
      debugFeedbackPanel.classList.remove("hidden");
    } else {
      debugFeedbackPanel.classList.add("hidden");
    }
  }

  // Fetch initial feedback stats
  fetchFeedbackStats();

  async function fetchFeedbackStats() {
    try {
      const res = await fetch("/api/feedback-stats");
      if (res.ok) {
        const json = await res.json();
        feedbackCounterBadge.textContent = `Feedback Learned: ${json.total_feedback || 0}`;
      }
    } catch (e) {
      console.warn("Could not fetch feedback stats", e);
    }
  }

  // Star Rating Picker interaction
  starPickBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const val = parseInt(btn.dataset.val, 10);
      setFeedbackRating(val);
    });
  });

  function setFeedbackRating(val) {
    selectedFeedbackRating = val;
    starPickerVal.textContent = `${val} / 5`;
    starPickBtns.forEach(b => {
      const bVal = parseInt(b.dataset.val, 10);
      if (bVal <= val) {
        b.classList.add("active");
      } else {
        b.classList.remove("active");
      }
    });

    // Auto sync sentiment pills
    if (val >= 4) {
      setFeedbackSentiment("Positive");
    } else if (val === 3) {
      setFeedbackSentiment("Neutral");
    } else {
      setFeedbackSentiment("Negative");
    }
  }

  // Sentiment Pill interaction
  pillPickBtns.forEach(pill => {
    pill.addEventListener("click", () => {
      setFeedbackSentiment(pill.dataset.sentiment);
      if (pill.dataset.sentiment === "Positive" && selectedFeedbackRating < 4) {
        setFeedbackRating(5);
      } else if (pill.dataset.sentiment === "Negative" && selectedFeedbackRating > 2) {
        setFeedbackRating(1);
      } else if (pill.dataset.sentiment === "Neutral") {
        setFeedbackRating(3);
      }
    });
  });

  function setFeedbackSentiment(sentiment) {
    selectedFeedbackSentiment = sentiment;
    pillPickBtns.forEach(p => {
      if (p.dataset.sentiment === sentiment) {
        p.classList.add("active");
      } else {
        p.classList.remove("active");
      }
    });
  }

  // "Teach Model" Live Feedback Submission
  teachModelBtn.addEventListener("click", async () => {
    if (!currentSingleReviewText) return;

    teachModelBtn.disabled = true;
    teachModelBtn.innerHTML = `<span>Updating model...</span>`;

    try {
      const response = await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          review: currentSingleReviewText,
          rating: selectedFeedbackRating,
          sentiment: selectedFeedbackSentiment
        })
      });

      const res = await response.json();
      if (!response.ok) {
        alert(res.error || "Failed to submit feedback.");
        return;
      }

      feedbackCounterBadge.textContent = `Feedback Learned: ${res.data.total_feedback_count}`;
      debugFeedbackMsg.textContent = `✓ Feedback learned! Model retrained live for "${selectedFeedbackSentiment}" (${selectedFeedbackRating}★).`;
      debugFeedbackMsg.classList.remove("hidden");

      setTimeout(() => {
        debugFeedbackMsg.classList.add("hidden");
      }, 4500);

      // Immediately display updated prediction from retrained model!
      if (res.data.updated_prediction) {
        displaySingleResult(res.data.updated_prediction);
      }
    } catch (err) {
      alert("Error teaching model: " + err.message);
    } finally {
      teachModelBtn.disabled = false;
      teachModelBtn.innerHTML = `<span>⚡ Teach Model</span>`;
    }
  });

  // ==========================================================================
  // Single Review Analysis
  // ==========================================================================

  unifiedForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = reviewInput.value.trim();
    if (!text) {
      reviewInput.focus();
      return;
    }

    currentSingleReviewText = text;
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = `<span>Analyzing...</span>`;

    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ review: text })
      });

      const res = await response.json();
      if (!response.ok) {
        alert(res.error || "Failed to analyze review.");
        return;
      }

      displaySingleResult(res.data);
      // Scroll smoothly to single result
      singleResultContainer.scrollIntoView({ behavior: "smooth", block: "nearest" });
    } catch (err) {
      alert("Error connecting to backend: " + err.message);
    } finally {
      analyzeBtn.disabled = false;
      analyzeBtn.innerHTML = `
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="11" cy="11" r="8"></circle>
          <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
        </svg>
        <span>Analyze Sentiment</span>
      `;
    }
  });

  function displaySingleResult(data) {
    singleResultContainer.classList.remove("hidden");

    // Verdict Badge
    resultSentimentBadge.textContent = data.sentiment;
    resultSentimentBadge.className = `sentiment-badge ${getBadgeClass(data.sentiment)}`;

    // Stars
    resultStars.textContent = renderStars(data.stars);
    resultStarsVal.textContent = `${data.stars.toFixed(1)} / 5.0`;

    // Polarity
    polarityScoreVal.textContent = `Polarity: ${formatPolarity(data.polarity)}`;
    const fillPercent = ((data.polarity + 1.0) / 2.0) * 100;
    polarityFill.style.width = `${Math.max(5, Math.min(95, fillPercent))}%`;

    // Probabilities
    resultConfidence.textContent = `${data.confidence.toFixed(1)}%`;
    probPositive.textContent = `${data.probabilities.Positive.toFixed(1)}%`;
    probNeutral.textContent = `${data.probabilities.Neutral.toFixed(1)}%`;
    probNegative.textContent = `${data.probabilities.Negative.toFixed(1)}%`;

    // Cues
    const posCues = data.key_cues?.positive_cues || [];
    const negCues = data.key_cues?.negative_cues || [];
    keyCuesList.innerHTML = "";

    if (posCues.length > 0 || negCues.length > 0) {
      keyCuesSection.classList.remove("hidden");
      posCues.forEach(c => {
        const tag = document.createElement("span");
        tag.className = "cue-tag cue-positive";
        tag.textContent = `+ ${c}`;
        keyCuesList.appendChild(tag);
      });
      negCues.forEach(c => {
        const tag = document.createElement("span");
        tag.className = "cue-tag cue-negative";
        tag.textContent = `- ${c}`;
        keyCuesList.appendChild(tag);
      });
    } else {
      keyCuesSection.classList.add("hidden");
    }

    // Initialize debug feedback rating to match current prediction
    setFeedbackRating(Math.round(data.stars));
    setFeedbackSentiment(data.sentiment);
    syncDebugPanelVisibility();
  }

  // Quick sample chips
  sampleChips.forEach(chip => {
    chip.addEventListener("click", () => {
      reviewInput.value = chip.dataset.sample;
      unifiedForm.dispatchEvent(new Event("submit"));
    });
  });

  // Clear button
  clearBtn.addEventListener("click", () => {
    reviewInput.value = "";
    currentSingleReviewText = "";
    singleResultContainer.classList.add("hidden");
    batchSection.classList.add("hidden");
    fileInput.value = "";
  });

  // ==========================================================================
  // Instant File Upload Handling (Combined into same box)
  // ==========================================================================

  browseBtn.addEventListener("click", () => {
    fileInput.click();
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      instantProcessFile(e.target.files[0]);
    }
  });

  // Drag and drop directly onto the unified box
  ["dragenter", "dragover"].forEach(name => {
    unifiedCard.addEventListener(name, (e) => {
      e.preventDefault();
      unifiedCard.classList.add("drag-active");
    });
  });

  ["dragleave", "drop"].forEach(name => {
    unifiedCard.addEventListener(name, (e) => {
      e.preventDefault();
      unifiedCard.classList.remove("drag-active");
    });
  });

  unifiedCard.addEventListener("drop", (e) => {
    if (e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      if (file.name.toLowerCase().endsWith(".txt")) {
        instantProcessFile(file);
      } else {
        alert("Please drop a .txt review file.");
      }
    }
  });

  // Load sample reviews directly & immediately open batch analysis
  loadSampleBtn.addEventListener("click", async () => {
    try {
      loadSampleBtn.disabled = true;
      loadSampleBtn.innerHTML = `<span>Loading...</span>`;
      const resp = await fetch("/api/sample-txt");
      if (!resp.ok) throw new Error("Could not fetch sample reviews");
      const text = await resp.text();
      const sampleBlob = new Blob([text], { type: "text/plain" });
      const sampleFile = new File([sampleBlob], "sample_reviews.txt", { type: "text/plain" });
      instantProcessFile(sampleFile);
    } catch (err) {
      alert("Error loading sample reviews: " + err.message);
    } finally {
      loadSampleBtn.disabled = false;
      loadSampleBtn.innerHTML = `<span>Load Sample .txt</span>`;
    }
  });

  async function instantProcessFile(file) {
    if (!file.name.toLowerCase().endsWith(".txt")) {
      alert("Please upload a .txt file containing reviews on separate lines.");
      return;
    }

    // Hide single result, show loading indicator and open file analysis box
    singleResultContainer.classList.add("hidden");
    batchSection.classList.remove("hidden");
    fileLoadingIndicator.classList.remove("hidden");
    batchFilenameDisplay.textContent = `Analyzing ${file.name}...`;
    batchSection.scrollIntoView({ behavior: "smooth", block: "start" });

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("/api/analyze-file", {
        method: "POST",
        body: formData
      });

      const res = await response.json();
      if (!response.ok) {
        alert(res.error || "Failed to process file.");
        batchSection.classList.add("hidden");
        return;
      }

      currentBatchData = res.data;
      displayBatchResults(res.data);
    } catch (err) {
      alert("Error processing review file: " + err.message);
      batchSection.classList.add("hidden");
    } finally {
      fileLoadingIndicator.classList.add("hidden");
      fileInput.value = "";
    }
  }

  // ==========================================================================
  // Render Batch Results & Line-by-Line Table
  // ==========================================================================

  function displayBatchResults(data) {
    batchSection.classList.remove("hidden");
    const summary = data.summary;

    batchFilenameDisplay.textContent = `Results for file: ${data.filename || "Uploaded File"}`;

    // Summary Cards
    metricTotalCount.textContent = summary.total_reviews;

    // Overall Average Sentiment
    metricOverallSentiment.textContent = summary.overall_sentiment;
    metricOverallSentiment.className = `sentiment-badge badge-lg ${getBadgeClass(summary.overall_sentiment)}`;
    metricAvgPolarity.textContent = `Avg Polarity: ${formatPolarity(summary.average_polarity)} | ${summary.average_stars} ★`;

    // Counts & Percentages
    metricPosCount.textContent = summary.positive_count;
    metricPosPct.textContent = `${summary.positive_percentage}% of total`;

    metricNeuCount.textContent = summary.neutral_count;
    metricNeuPct.textContent = `${summary.neutral_percentage}% of total`;

    metricNegCount.textContent = summary.negative_count;
    metricNegPct.textContent = `${summary.negative_percentage}% of total`;

    // Proportional Bar
    distBarPos.style.width = `${summary.positive_percentage}%`;
    distBarNeu.style.width = `${summary.neutral_percentage}%`;
    distBarNeg.style.width = `${summary.negative_percentage}%`;

    // Filter pill counters
    filterCountAll.textContent = summary.total_reviews;
    filterCountPos.textContent = summary.positive_count;
    filterCountNeu.textContent = summary.neutral_count;
    filterCountNeg.textContent = summary.negative_count;

    renderTableRows();
  }

  function renderTableRows() {
    if (!currentBatchData || !currentBatchData.reviews) return;

    const searchTerm = tableSearch.value.toLowerCase().trim();
    reviewsTbody.innerHTML = "";

    const filtered = currentBatchData.reviews.filter(item => {
      const matchesFilter = (activeFilter === "all") || (item.sentiment === activeFilter);
      const matchesSearch = !searchTerm || item.text.toLowerCase().includes(searchTerm);
      return matchesFilter && matchesSearch;
    });

    if (filtered.length === 0) {
      const emptyTr = document.createElement("tr");
      emptyTr.innerHTML = `
        <td colspan="6" style="text-align: center; padding: 2rem; color: var(--text-muted);">
          No reviews matched the selected filter or search term.
        </td>
      `;
      reviewsTbody.appendChild(emptyTr);
      return;
    }

    filtered.forEach(item => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><span class="line-num-badge">#${item.line_number}</span></td>
        <td class="review-cell-text">${escapeHtml(item.text)}</td>
        <td><span class="sentiment-badge ${getBadgeClass(item.sentiment)}">${item.sentiment}</span></td>
        <td><strong>${formatPolarity(item.polarity)}</strong></td>
        <td><span style="color: #f59e0b;">${renderStars(item.stars)}</span> <small>(${item.stars})</small></td>
        <td>${item.confidence.toFixed(1)}%</td>
      `;
      reviewsTbody.appendChild(tr);
    });
  }

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  // Filter pills
  filterPills.forEach(pill => {
    pill.addEventListener("click", () => {
      filterPills.forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      activeFilter = pill.dataset.filter;
      renderTableRows();
    });
  });

  // Search input
  tableSearch.addEventListener("input", () => {
    renderTableRows();
  });

  // CSV Export
  exportCsvBtn.addEventListener("click", () => {
    if (!currentBatchData || !currentBatchData.reviews) return;

    let csvContent = "data:text/csv;charset=utf-8,";
    csvContent += "LineNumber,Sentiment,Polarity,Stars,Confidence,ReviewText\n";

    currentBatchData.reviews.forEach(row => {
      const cleanReview = `"${row.text.replace(/"/g, '""')}"`;
      csvContent += `${row.line_number},${row.sentiment},${row.polarity},${row.stars},${row.confidence}%,${cleanReview}\n`;
    });

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `sentiment_analysis_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  });
});
