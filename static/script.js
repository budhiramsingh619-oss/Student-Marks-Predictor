
document.addEventListener("DOMContentLoaded", function () {

    const form = document.querySelector("form");

    if (form) {
        form.addEventListener("submit", function () {

            const button = form.querySelector(".predict-btn");

            button.textContent = "Analyzing Student Performance...";
            button.disabled = true;

        });
    }

});
