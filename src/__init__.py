"""
<style>
/* Ocultar el título 'src' y el botón de 'View Source' de pdoc */
h1.modulename, .view-source-button {
    display: none !important;
}

/* Eliminar los márgenes estrechos de pdoc para aprovechar la pantalla */
@media (min-width: 770px) {
    main.pdoc {
        width: 100% !important;
        padding: 0 0 0 var(--sidebar-width) !important;
    }
}

/* Ajustar el iframe para que ocupe todo el espacio disponible */
.notebook-embed {
    width: 100%;
    height: 100vh;
    border: none;
    display: block;
}
</style>

<!-- Cargar el HTML del notebook independiente CON LOS PERMISOS DE NAVEGACIÓN -->
<iframe class="notebook-embed" src="./index.html" sandbox="allow-scripts allow-same-origin allow-top-navigation allow-top-navigation-by-user-activation" title="Jupyter Notebook"></iframe>
"""
