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

// function setVisibility(el, on) {
//   const hidden_element = "hidden_element";
//   if (on & el.classList.contains(hidden_element)) {
//     el.classList.remove(hidden_element);
//   } else if (!on) {
//     el.classList.add(hidden_element);
//   }
// }

function setVisibilityForDropdowns(el, on) {
  if (on) {
    el.style.display = "block";
    el.style.height = "20px";
    el.style.width = "100%";
  } else {
    el.style.display = "none";
    el.style.height = "0px";
    el.style.width = "0px";
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
  const customVaeDropdown = gradioApp().querySelector(
    "div[id$=custom_vae_choice]"
  );

  switch (modelTypeDropdown.value) {
    case "Installed Models":
      setVisibilityForDropdowns(downloadModelDropdown, false);
      setVisibilityForDropdowns(cachedModelDropdown, true);
      setVisibilityForDropdowns(customModelDropdown, false);
      setVisibilityForDropdowns(customVaeDropdown, false);
      break;
    case "Downloadable Models":
      setVisibilityForDropdowns(downloadModelDropdown, true);
      setVisibilityForDropdowns(cachedModelDropdown, false);
      setVisibilityForDropdowns(customModelDropdown, false);
      setVisibilityForDropdowns(customVaeDropdown, false);
      break;
    case "Custom Models":
      setVisibilityForDropdowns(downloadModelDropdown, false);
      setVisibilityForDropdowns(cachedModelDropdown, false);
      setVisibilityForDropdowns(customModelDropdown, true);
      setVisibilityForDropdowns(customVaeDropdown, false);
      break;
    case "Load Custom Vae To Current Model":
      setVisibilityForDropdowns(downloadModelDropdown, false);
      setVisibilityForDropdowns(cachedModelDropdown, false);
      setVisibilityForDropdowns(customModelDropdown, false);
      setVisibilityForDropdowns(customVaeDropdown, true);
      break;
  }
}

// function runOnStart() {
//   const downloadModelDropdown = gradioApp().querySelector(
//     "div[id$=download_model_choice]"
//   );
//   const cachedModelDropdown = gradioApp().querySelector(
//     "div[id$=cached_model_choice]"
//   );
//   const customModelDropdown = gradioApp().querySelector(
//     "div[id$=custom_model_choice]"
//   );

//   setVisibilityForDropdowns(downloadModelDropdown, false);
//   setVisibilityForDropdowns(cachedModelDropdown, false);
//   setVisibilityForDropdowns(customModelDropdown, false);
// }

// // run startup javascript after elements load in
// setTimeout(() => {
//   runOnStart();
// }, 100);
// // do it again
// setTimeout(() => {
//   runOnStart();
// }, 1000);
