// Form elements
const form = document.getElementById('changeRequestForm');
const titleInput = document.getElementById('title');
const descInput = document.getElementById('description');
const commentsInput = document.getElementById('comments');
const filenameInput = document.getElementById('filename');
const titleError = document.getElementById('titleError');
const descError = document.getElementById('descriptionError');
const loadingIndicator = document.getElementById('loadingIndicator');
const clearBtn = document.getElementById('clearForm');
const generateSampleBtn = document.getElementById('generateSample');
const kindBug = document.getElementById('kindBug');
const kindFeature = document.getElementById('kindFeature');
const kindError = document.getElementById('kindError');

// Auto-save draft
const DRAFT_KEY = 'k8s-change-request-draft';

function saveDraft() {
  let kind = '';
  if (kindBug.checked) kind = 'bug';
  if (kindFeature.checked) kind = 'feature';
  localStorage.setItem(DRAFT_KEY, JSON.stringify({
    title: titleInput.value,
    description: descInput.value,
    comments: commentsInput.value,
    filename: filenameInput.value,
    kind
  }));
}

function loadDraft() {
  const draft = localStorage.getItem(DRAFT_KEY);
  if (draft) {
    const { title, description, comments, filename, kind } = JSON.parse(draft);
    titleInput.value = title || '';
    descInput.value = description || '';
    commentsInput.value = comments || '';
    filenameInput.value = filename || '';
    kindBug.checked = kind === 'bug';
    kindFeature.checked = kind === 'feature';
  }
}

titleInput.addEventListener('input', saveDraft);
descInput.addEventListener('input', saveDraft);
commentsInput.addEventListener('input', saveDraft);
filenameInput.addEventListener('input', saveDraft);
kindBug.addEventListener('change', saveDraft);
kindFeature.addEventListener('change', saveDraft);
window.addEventListener('DOMContentLoaded', loadDraft);

// Form validation
function validateForm() {
  let valid = true;
  if (!titleInput.value.trim()) {
    titleError.classList.remove('hidden');
    valid = false;
  } else {
    titleError.classList.add('hidden');
  }
  if (!descInput.value.trim()) {
    descError.classList.remove('hidden');
    valid = false;
  } else {
    descError.classList.add('hidden');
  }
  if (!kindBug.checked && !kindFeature.checked) {
    kindError.classList.remove('hidden');
    valid = false;
  } else {
    kindError.classList.add('hidden');
  }
  return valid;
}

// Clear form
clearBtn.addEventListener('click', () => {
  titleInput.value = '';
  descInput.value = '';
  commentsInput.value = '';
  filenameInput.value = '';
  kindBug.checked = false;
  kindFeature.checked = false;
  titleError.classList.add('hidden');
  descError.classList.add('hidden');
  kindError.classList.add('hidden');
  localStorage.removeItem(DRAFT_KEY);
  // Hide classification result and similar requests table
  document.getElementById('classificationResultSection').classList.add('hidden');
  document.getElementById('similarRequestsSection').classList.add('hidden');
});

// Function to reset the loading indicator to its initial state
function resetLoadingIndicator() {
  loadingIndicator.classList.add('hidden');
  const progressBar = document.getElementById('progressBar');
  const progressText = document.getElementById('progressText');
  progressBar.style.width = '0%';
  progressText.textContent = 'Processing...';
}

// Global state for pagination
let currentPage = 1;
const pageSize = 5;
let similarRequests = [];

// Form submission with AJAX
form.addEventListener('submit', (e) => {
  e.preventDefault();
  if (!validateForm()) return;
  
  loadingIndicator.classList.remove('hidden');
  const progressBar = document.getElementById('progressBar');
  const progressText = document.getElementById('progressText');
  let progress = 0;
  progressBar.style.width = '0%';
  progressText.textContent = 'Processing...';
  
  // Prepare form data
  const formData = new FormData();
  formData.append('title', titleInput.value);
  formData.append('description', descInput.value);
  formData.append('comments', commentsInput.value);
  formData.append('kind', kindBug.checked ? 'bug' : 'feature');
  formData.append('filename', filenameInput.value);  // Add filename to form data
  
  // Simulate progress during API call
  const progressInterval = setInterval(() => {
    progress += Math.floor(Math.random() * 18) + 7; // random step for realism
    if (progress >= 95) progress = 95; // cap at 95% until we get the response
    progressBar.style.width = progress + '%';
    progressText.textContent = `Processing... ${progress}%`;
  }, 120);
  
  // Send AJAX request to Flask backend
  fetch('/classify', {
    method: 'POST',
    body: formData
  })
  .then(response => response.json())
  .then(data => {
    clearInterval(progressInterval);
    
    if (data.success) {
      // Complete the progress bar
      progress = 100;
      progressBar.style.width = '100%';
      progressText.textContent = 'Processing... 100%';
      
      // Display classification results
      displayClassificationResult(data.classification);
      
      // Save similar requests for pagination
      similarRequests = data.similar_requests;
      
      // Update area filter dropdown with available areas
      updateAreaFilterOptions();
      
      // Show results sections
      setTimeout(() => {
        loadingIndicator.classList.add('hidden');
        document.getElementById('classificationResultSection').classList.remove('hidden');
        document.getElementById('similarRequestsSection').classList.remove('hidden');
        
        // Reset pagination and populate similar requests
        currentPage = 1;
        populateSimilarRequests(currentPage);
      }, 500); // Increased timeout to ensure UI updates complete
    } else {
      // Handle error
      alert('Error processing your request. Please try again.');
      resetLoadingIndicator();
    }
  })
  .catch(error => {
    clearInterval(progressInterval);
    console.error('Error:', error);
    alert('Error processing your request. Please try again.');
    resetLoadingIndicator();
  });
});

// Display classification result
function displayClassificationResult(classification) {
  // Set timestamp
  document.getElementById('timestamp').textContent = classification.timestamp;
  
  // Display model type information in its dedicated element
  if (classification.model_type) {
    document.getElementById('modelTypeInfo').innerHTML = classification.model_type;
    document.getElementById('modelTypeInfo').classList.remove('hidden');
  } else {
    document.getElementById('modelTypeInfo').classList.add('hidden');
  }
  
  // Set classification reasoning (without the model type)
  document.getElementById('classificationReasoning').innerHTML = classification.reasoning;
  
  // Clear and populate area labels
  const areaLabelsContainer = document.getElementById('areaLabels');
  const timestampElement = document.getElementById('timestamp');

  // Remove all existing area labels (but keep the timestamp)
  Array.from(areaLabelsContainer.children).forEach(child => {
    if (child !== timestampElement) {
      child.remove();
    }
  });


  // If no labels/areas are predicted, show message
  if (!classification.areas || classification.areas.length === 0) {
    // Show message in areaLabels
    const msg = document.createElement('span');
    msg.className = 'text-red-600 text-sm font-semibold';
    msg.textContent = 'No area labels could be predicted for this request.';
    areaLabelsContainer.insertBefore(msg, timestampElement);
    return;
  }

  // Add area badges
  classification.areas.forEach(area => {
    const badge = document.createElement('span');
    // Use the color provided from the backend
    const colorClass = area.color || 'bg-gray-100 text-gray-800'; // Use backend color or default
    badge.className = `inline-block px-3 py-1 rounded-full text-sm font-semibold mr-2 ${colorClass}`;
    badge.innerHTML = `${area.name} <span class="ml-1 font-normal">${area.confidence}%</span>`;
    // Insert the badge before the timestamp
    areaLabelsContainer.insertBefore(badge, timestampElement);
  });
}

// Populate similar requests table
function populateSimilarRequests(page = 1) {
  const tbody = document.getElementById('similarRequestsTable');
  tbody.innerHTML = '';
  
  // Calculate pagination indices
  const startIdx = (page - 1) * pageSize;
  const endIdx = Math.min(startIdx + pageSize, similarRequests.length);
  
  // Get current filter area
  const areaFilter = document.getElementById('filterArea').value;
  
  // Filter requests by area if needed
  let filteredRequests = similarRequests;
  if (areaFilter) {
    filteredRequests = similarRequests.filter(req => 
      req.areas.some(area => area.name === areaFilter)
    );
  }
  
  // Get the slice for current page
  const pageRequests = filteredRequests.slice(startIdx, endIdx);
  
  // Populate table
  for (let i = 0; i < pageRequests.length; i++) {
    const req = pageRequests[i];
    const tr = document.createElement('tr');
    tr.className = 'hover:bg-blue-50 focus-within:bg-blue-100';
    
    // Format the similarity score to 2 decimal places
    const similarityScore = typeof req.similarity === 'number' 
      ? req.similarity.toFixed(2) + '%' 
      : req.similarity + '%';
    
    // Get a shortened description
    const shortDescription = req.description && req.description.length > 40 
      ? `${req.description.slice(0, 40)}...` 
      : (req.description || 'No description available');
    
    // Format the row HTML
    tr.innerHTML = `
      <td class="p-2">${startIdx + i + 1}</td>
      <td class="p-2 font-bold text-blue-700">${similarityScore}</td>
      <td class="p-2">${req.title || 'No title'}</td>
      <td class="p-2 truncate max-w-xs">
        <span class="desc-short">${shortDescription}</span>
        <button class="expandDesc text-blue-600 underline text-xs ml-1" data-idx="${startIdx + i}">Expand</button>
        <span class="desc-full hidden">${req.description || 'No description available'}</span>
      </td>
      <td class="p-2">${(req.areas || []).map(area => 
        `<span class='inline-block px-2 py-1 rounded-full ${area.color || 'bg-gray-100 text-gray-800'} text-xs mr-2 mb-1'>${area.name}</span>`
      ).join('')}</td>
      <td class="p-2">
        <button class="detailsBtn text-blue-700 underline text-xs" data-idx="${startIdx + i}">View</button>
      </td>
    `;
    tbody.appendChild(tr);
  }
  
  // Update pagination info
  document.getElementById('paginationInfo').textContent = `Showing ${filteredRequests.length > 0 ? startIdx + 1 : 0}-${endIdx} of ${filteredRequests.length}`;
  
  // Update button states
  document.getElementById('prevPageBtn').disabled = page === 1;
  document.getElementById('nextPageBtn').disabled = endIdx >= filteredRequests.length;
  
  // Update the area filter options
  updateAreaFilterOptions();
}

// Dynamically populate filter dropdown based on available areas
function updateAreaFilterOptions() {
  const areaFilter = document.getElementById('filterArea');
  // Get the currently selected value
  const currentValue = areaFilter.value;
  
  // Clear current options except the first "All" option
  while (areaFilter.options.length > 1) {
    areaFilter.options.remove(1);
  }
  
  // Add an entry to track which areas we've added to avoid duplicates
  const addedAreas = new Set();
  
  // Go through all similar requests and collect unique area names
  similarRequests.forEach(req => {
    (req.areas || []).forEach(area => {
      if (!addedAreas.has(area.name)) {
        addedAreas.add(area.name);
        const option = document.createElement('option');
        option.value = area.name;
        option.text = area.name;
        areaFilter.add(option);
      }
    });
  });
  
  // Try to restore the previously selected value if it exists in the new options
  if (currentValue && Array.from(areaFilter.options).some(opt => opt.value === currentValue)) {
    areaFilter.value = currentValue;
  }
}

// Filter area change event
document.getElementById('filterArea').addEventListener('change', () => {
  currentPage = 1; // Reset to first page when filter changes
  populateSimilarRequests(currentPage);
});

// Pagination button events
document.getElementById('prevPageBtn').addEventListener('click', () => {
  if (currentPage > 1) {
    currentPage--;
    populateSimilarRequests(currentPage);
  }
});

document.getElementById('nextPageBtn').addEventListener('click', () => {
  const areaFilter = document.getElementById('filterArea').value;
  let filteredRequests = similarRequests;
  
  if (areaFilter) {
    filteredRequests = similarRequests.filter(req => 
      req.areas.some(area => area.name === areaFilter)
    );
  }
  
  if (currentPage * pageSize < filteredRequests.length) {
    currentPage++;
    populateSimilarRequests(currentPage);
  }
});

// Expand/collapse description
document.getElementById('similarRequestsTable').addEventListener('click', (e) => {
  if (e.target.classList.contains('expandDesc')) {
    const td = e.target.closest('td');
    td.querySelector('.desc-short').classList.toggle('hidden');
    td.querySelector('.desc-full').classList.toggle('hidden');
    e.target.textContent = td.querySelector('.desc-full').classList.contains('hidden') ? 'Expand' : 'Collapse';
  }
  if (e.target.classList.contains('detailsBtn')) {
    const idx = parseInt(e.target.getAttribute('data-idx'));
    showDetailModal(idx);
  }
});

// Detail modal logic
const detailModal = document.getElementById('detailModal');
const closeModalBtn = document.getElementById('closeModalBtn');

// Add event listener for the bottom close button
document.getElementById('closeModalBtnBottom').addEventListener('click', () => {
  detailModal.classList.add('hidden');
});

closeModalBtn.addEventListener('click', () => {
  detailModal.classList.add('hidden');
});

function showDetailModal(idx) {
  const req = similarRequests[idx];
  
  // Store the current request data at the time of viewing
  const currentTitle = titleInput.value;
  const currentDescription = descInput.value;
  const currentComments = commentsInput.value;
  const currentFilename = filenameInput.value;
  const currentType = kindBug.checked ? 'Bug' : (kindFeature.checked ? 'Feature Request' : 'Not specified');
  
  // Format URLs to be more readable
  const formatUrl = (url) => {
    if (!url) return '';
    // If URL is too long, truncate for display but keep full URL in href
    const displayUrl = url.length > 40 ? url.substring(0, 35) + '...' : url;
    return `<a href="${url}" target="_blank" class="ml-1 text-blue-600 underline text-sm">${displayUrl}</a>`;
  };
  
  // Format descriptions to handle long content
  const formatDescription = (desc) => {
    if (!desc) return 'No description available';
    // Convert URLs to clickable links
    return desc.replace(
      /(https?:\/\/[^\s]+)/g, 
      '<a href="$1" target="_blank" class="text-blue-600 underline">$1</a>'
    );
  };
  
  const selectedRequestHtml = `
    <div class="mb-3">
      <div class="font-medium text-gray-700">Title:</div>
      <div class="ml-1 mb-2">${req.title || 'No title available'}</div>
    </div>
    
    <div class="mb-3">
      <div class="font-medium text-gray-700">Description:</div>
      <div class="ml-1 mb-2 text-sm text-gray-800">${formatDescription(req.description)}</div>
    </div>

    ${req.issue_url ? 
      `<div class="mb-3">
        <div class="font-medium text-gray-700">GitHub Issue:</div>
        ${formatUrl(req.issue_url)}
      </div>` : ''}
    
    ${req.filename ? 
      `<div class="mb-3">
        <div class="font-medium text-gray-700">Filename:</div>
        <div class="ml-1 text-sm text-gray-800">${req.filename}</div>
      </div>` : ''}
    
    ${req.areas && req.areas.length > 0 ? 
      `<div class="mb-3">
        <div class="font-medium text-gray-700">Areas:</div>
        <div class="ml-1 flex flex-wrap gap-2 mt-1">
          ${req.areas.map(area => 
            `<span class='inline-block px-2 py-1 rounded-full ${area.color || 'bg-gray-100 text-gray-800'} text-xs'>${area.name}</span>`
          ).join('')}
        </div>
      </div>` : ''}
    
    <div class="mb-1">
      <div class="font-medium text-gray-700">Similarity Score:</div>
      <div class="ml-1 font-bold text-blue-700">${typeof req.similarity === 'number' ? req.similarity.toFixed(2) : req.similarity}%</div>
    </div>
  `;
  
  const currentRequestHtml = `
    <div class="mb-3">
      <div class="font-medium text-gray-700">Title:</div>
      <div class="ml-1 mb-2">${currentTitle || 'Not specified'}</div>
    </div>
    
    <div class="mb-3">
      <div class="font-medium text-gray-700">Description:</div>
      <div class="ml-1 mb-2 text-sm text-gray-800">${formatDescription(currentDescription)}</div>
    </div>
    
    ${currentComments ? 
      `<div class="mb-3">
        <div class="font-medium text-gray-700">Comments:</div>
        <div class="ml-1 mb-2 text-sm text-gray-800">${formatDescription(currentComments)}</div>
      </div>` : ''}
    
    ${currentFilename ? 
      `<div class="mb-3">
        <div class="font-medium text-gray-700">Filename:</div>
        <div class="ml-1 text-sm text-gray-800">${currentFilename}</div>
      </div>` : ''}
      
    <div class="mb-1">
      <div class="font-medium text-gray-700">Type:</div>
      <div class="ml-1 text-sm text-gray-800">${currentType}</div>
    </div>
  `;
  
  document.getElementById('modalSelectedRequest').innerHTML = selectedRequestHtml;
  document.getElementById('modalCurrentRequest').innerHTML = currentRequestHtml;
  detailModal.classList.remove('hidden');
  
  document.getElementById('useAsTemplateBtn').onclick = () => {
    titleInput.value = req.title || '';
    descInput.value = req.description || '';
    // We don't set filename from historical request templates
    detailModal.classList.add('hidden');
    saveDraft();
  };
}

// Accessibility: close modal with Escape
window.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') detailModal.classList.add('hidden');
});

// Generate random sample from testing data
generateSampleBtn.addEventListener('click', function() {
  // Indicate loading state
  generateSampleBtn.classList.add('opacity-70');
  generateSampleBtn.innerHTML = 'Loading...';
  
  // Get the specified filename if any
  const requestedFilename = filenameInput.value.trim();
  
  // Get the currently selected request type (bug or feature)
  // If neither is selected, default to bug
  const requestType = kindBug.checked ? 'bug' : (kindFeature.checked ? 'feature' : 'bug');
  
  // Build the query string with filename and type
  let queryParams = [];
  if (requestedFilename) {
    queryParams.push(`filename=${encodeURIComponent(requestedFilename)}`);
  }
  queryParams.push(`type=${requestType}`);
  
  // Show which dataset is being used
  const toastMessage = document.createElement('div');
  toastMessage.className = 'fixed bottom-4 right-4 bg-green-100 text-green-800 px-4 py-2 rounded shadow';
  toastMessage.innerText = `Generating random ${requestType} sample...`;
  document.body.appendChild(toastMessage);
  setTimeout(() => {
    toastMessage.remove();
  }, 3000);
  
  // Fetch a random sample from the appropriate testing_data.csv
  fetch('/random_sample?' + queryParams.join('&'))
    .then(response => {
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      return response.json();
    })
    .then(data => {
      if (data.success) {
        // Fill form with the random sample
        titleInput.value = data.sample.title || '';
        descInput.value = data.sample.description || '';
        commentsInput.value = data.sample.comments || '';
        
        // Set the filename if returned
        if (data.sample.filename) {
          filenameInput.value = data.sample.filename;
        }
        
        // Set the kind (bug/feature)
        if (data.sample.kind === 'bug') {
          kindBug.checked = true;
          kindFeature.checked = false;
        } else if (data.sample.kind === 'feature') {
          kindBug.checked = false;
          kindFeature.checked = true;
        }
        
        // Clear any existing error messages
        titleError.classList.add('hidden');
        descError.classList.add('hidden');
        kindError.classList.add('hidden');
        
        // Hide classification result and similar requests table
        document.getElementById('classificationResultSection').classList.add('hidden');
        document.getElementById('similarRequestsSection').classList.add('hidden');
        
        // Save to draft storage
        saveDraft();
      } else {
        console.error('Server returned error:', data.error);
        alert('Error loading sample: ' + (data.error || 'Unknown error'));
      }
    })
    .catch(error => {
      console.error('Error in fetch:', error.message);
      alert('Failed to load sample: ' + error.message);
    })
    .finally(() => {
      // Restore button state
      generateSampleBtn.classList.remove('opacity-70');
      generateSampleBtn.innerHTML = 'Generate Sample';
    });
});
