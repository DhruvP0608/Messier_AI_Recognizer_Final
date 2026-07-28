const catalogGrid = document.getElementById("catalogGrid");
const catalogSearch = document.getElementById("catalogSearch");
const catalogFilter = document.getElementById("catalogFilter");
const catalogCount = document.getElementById("catalogCount");

const detailBadge = document.getElementById("detailBadge");
const catalogDetailTitle = document.getElementById("catalogDetailTitle");
const catalogDetailDescription = document.getElementById("catalogDetailDescription");
const catalogDetailId = document.getElementById("catalogDetailId");
const catalogDetailCategory = document.getElementById("catalogDetailCategory");
const catalogReferenceImage = document.getElementById("catalogReferenceImage");
const catalogStarMapImage = document.getElementById("catalogStarMapImage");

let catalogEntries = [];
let selectedId = null;

function renderDetail(entry) {
  selectedId = entry.messier_id;
  detailBadge.textContent = entry.common_name || "Messier Object";
  catalogDetailTitle.textContent = entry.title;
  catalogDetailDescription.textContent = entry.description;
  catalogDetailId.textContent = entry.messier_id;
  catalogDetailCategory.textContent = entry.category;
  catalogReferenceImage.src = entry.reference_image;
  catalogStarMapImage.src = entry.star_map_image || entry.reference_image;
  catalogReferenceImage.alt = `${entry.title} reference image`;
  catalogStarMapImage.alt = `${entry.title} star map`;
  renderGrid();
}

function getFilteredEntries() {
  const query = catalogSearch.value.trim().toLowerCase();
  const category = catalogFilter.value;

  return catalogEntries.filter((entry) => {
    const matchesCategory = category === "All" || entry.category === category;
    const haystack = [
      entry.messier_id,
      entry.title,
      entry.common_name || "",
      entry.category,
      entry.description,
    ]
      .join(" ")
      .toLowerCase();
    const matchesSearch = !query || haystack.includes(query);
    return matchesCategory && matchesSearch;
  });
}

function renderGrid() {
  const filtered = getFilteredEntries();
  catalogCount.textContent = `${filtered.length} objects`;

  catalogGrid.innerHTML = filtered
    .map(
      (entry) => `
        <button class="catalog-card ${entry.messier_id === selectedId ? "active" : ""}" data-id="${entry.messier_id}" type="button">
          <div class="catalog-thumb">
            <img src="${entry.reference_image}" alt="${entry.title}" />
          </div>
          <div class="catalog-card-body">
            <span class="catalog-card-id">${entry.messier_id}</span>
            <h4>${entry.title}</h4>
            <p>${entry.category}</p>
          </div>
        </button>
      `
    )
    .join("");

  for (const card of catalogGrid.querySelectorAll(".catalog-card")) {
    card.addEventListener("click", () => {
      const entry = catalogEntries.find((item) => item.messier_id === card.dataset.id);
      if (entry) {
        renderDetail(entry);
      }
    });
  }
}

function populateFilter(entries) {
  const categories = [...new Set(entries.map((entry) => entry.category))];
  for (const category of categories) {
    const option = document.createElement("option");
    option.value = category;
    option.textContent = category;
    catalogFilter.append(option);
  }
}

async function loadCatalog() {
  const response = await fetch("/api/catalog");
  const payload = await response.json();
  catalogEntries = payload.entries;
  populateFilter(catalogEntries);
  renderDetail(catalogEntries[0]);
}

catalogSearch.addEventListener("input", renderGrid);
catalogFilter.addEventListener("change", renderGrid);

loadCatalog().catch(() => {
  catalogDetailTitle.textContent = "Catalog unavailable";
  catalogDetailDescription.textContent = "The Messier catalog data could not be loaded.";
});

