document.addEventListener("DOMContentLoaded", () => {
    const overlay = document.getElementById("aslOverlay");
    const showBtn = document.getElementById("showASLBtn");
    const aslImage = overlay.querySelector(".asl-img");
    const aslText = document.getElementById("aslText");

    showBtn.addEventListener("click", () => {
        overlay.style.display = "flex";
        overlay.focus();

        function handleKey(e) {
            if (e.key.toLowerCase() === "s") {
                // Fetch rutu koja pokreće Python skriptu u pozadini pomoću Flask-a
                fetch("/learn_sign")
                    .then(response => console.log("Skripta pokrenuta u pozadini"))
                    .catch(err => console.error(err));

                // Uklanjanje teksta
                aslText.style.display = "none";

                // Pomeranje slike u gornji desni ugao
                aslImage.classList.remove("center-img");
                aslImage.classList.add("top-right");

                document.removeEventListener("keydown", handleKey);
            }
        }

        document.addEventListener("keydown", handleKey);
    });

    overlay.addEventListener("click", () => {
        overlay.style.display = "none";
    });
});
