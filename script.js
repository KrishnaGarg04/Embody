document.addEventListener('DOMContentLoaded', function() {
  const fontToggle = document.getElementById('font-toggle');
  const body = document.body;

  fontToggle.addEventListener('click', function() {
    body.classList.toggle('dyslexic-font');
    if (body.classList.contains('dyslexic-font')) {
      fontToggle.textContent = 'Default Font';
    } else {
      fontToggle.textContent = 'Dyslexic Font';
    }
  });
});

