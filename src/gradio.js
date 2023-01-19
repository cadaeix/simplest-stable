const gradioApp = () =>
  document.getElementsByTagName("gradio-app")[0].shadowRoot || document;

const selectedGalleryIndex = () =>
  Array.from(
    gradioApp().querySelectorAll("div[id$=output_gallery] .gallery-item")
  ).findIndex((v) => v.classList.contains("!ring-2"));

function findSelectedImageFromGallery(gallery) {
  if (gallery.length == 1) {
    return gallery[0];
  }

  const index = selectedGalleryIndex();
  return index != -1 ? gallery[index] : null;
}

function recreateNode(el, withChildren) {
  if (withChildren) {
    el.parentNode.replaceChild(el.cloneNode(true), el);
  } else {
    var newEl = el.cloneNode(false);
    while (el.hasChildNodes()) newEl.appendChild(el.firstChild);
    el.parentNode.replaceChild(newEl, el);
  }
}

function setVisibility(el, on) {
  const hidden_element = "hidden_element";
  if (on & el.classList.contains(hidden_element)) {
    el.classList.remove(hidden_element);
  } else if (!on) {
    el.classList.add(hidden_element);
  }
}

function handleModelDropdowns() {
  const modelTypeDropdown = gradioApp().querySelector(
    "div[id$=model_dropdown_type] label select"
  );

  const downloadModelDropdown = gradioApp().querySelector(
    "div[id$=download_model_choice]"
  );
  const cachedModelDropdown = gradioApp().querySelector(
    "div[id$=cached_model_choice]"
  );
  const customModelDropdown = gradioApp().querySelector(
    "div[id$=custom_model_choice]"
  );

  switch (modelTypeDropdown.value) {
    case "Installed Models":
      setVisibility(downloadModelDropdown, false);
      setVisibility(cachedModelDropdown, true);
      setVisibility(customModelDropdown, false);
      break;
    case "Downloadable Models":
      setVisibility(downloadModelDropdown, true);
      setVisibility(cachedModelDropdown, false);
      setVisibility(customModelDropdown, false);
      break;
    case "Custom Models":
      setVisibility(downloadModelDropdown, false);
      setVisibility(cachedModelDropdown, false);
      setVisibility(customModelDropdown, true);
      break;
  }
}

function runOnStart() {
  const downloadModelDropdown = gradioApp().querySelector(
    "div[id$=download_model_choice]"
  );
  const cachedModelDropdown = gradioApp().querySelector(
    "div[id$=cached_model_choice]"
  );
  const customModelDropdown = gradioApp().querySelector(
    "div[id$=custom_model_choice]"
  );

  setVisibility(downloadModelDropdown, true);
  setVisibility(cachedModelDropdown, false);
  setVisibility(customModelDropdown, false);
}

// run startup javascript after elements load in
setTimeout(() => {
  runOnStart();
}, 500);
