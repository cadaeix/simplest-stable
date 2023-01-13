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

// function setupOutputGallery() {
//   console.log("lol");
//   setTimeout(() => {
//     gradioApp()
//       .querySelectorAll("div[id$=output_gallery] .gallery-item")[0]
//       .click();
//   }, 100);
//   console.log("lol2");
//   setTimeout(() => {
//     gradioApp().querySelectorAll(".modify-upload.z-10.top-2")[0].remove();
//   }, 100);
//   console.log("lol3");
//   setTimeout(() => {
//     recreateNode(
//       gradioApp().querySelectorAll(
//         "#output_gallery > .absolute.group.inset-0"
//       )[0],
//       false
//     );
//   }, 100);
// }
