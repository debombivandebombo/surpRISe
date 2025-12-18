# 📋 Cheat List: Git & uv para surpRISe

Guía de referencia rápida para la gestión del simulador de **la RIS** en tu **portátil**.

---

## 1. Gestión de Git (Control de Versiones)

### Guardar cambios (Commit)
   ```bash
   # Añade todos los cambios (archivos nuevos, modificados y borrados)
   git add .

   # Registra los cambios con un mensaje descriptivo
   git commit -m "Tu mensaje descriptivo aquí"

   # Sube los cambios a GitHub
   git push origin uv-migration
   ```

### Crear una nueva rama de trabajo
   ```bash
   git checkout -b nombre-de-la-nueva-rama
   ```

## 2. Sincronización entre Equipos (Portátil ↔ PC)
### Flujo de trabajo al cambiar de equipo:

1. Actualizar información remota: ```git fetch```

2. Cambiar a la rama de trabajo: ```git checkout uv-migration```

3. Descargar cambios: ```git pull```

4. Sincronizar el entorno de Python: ```uv sync```

## 3. Generar una Nueva Versión (dist/)
### Pasos para actualizar la distribución de la librería:

1. Modificar versión: Abre pyproject.toml y edita version = "0.x.x".

2. Limpiar versiones antiguas (Windows): ```rmdir /s /q dist```

3. Construir el paquete: ```uv build```

## 4. Instalación en Otros Proyectos
### Opción A: Desde el archivo .whl (Versión Exacta)
Ideal para cuando las medidas y el código son estables.

Con uv: ```uv add "C:/ruta/a/surpRISe/dist/surpRISe-0.1.0-py3-none-any.whl"```

Con pip: ```pip install "C:/ruta/a/surpRISe/dist/surpRISe-0.1.0-py3-none-any.whl"```

### Opción B: Modo Editable (Desarrollo en tiempo real)
Ideal para ver cambios al instante sin necesidad de hacer uv build.

Con uv: ```uv add --editable "C:/ruta/a/surpRISe"```

Con pip: ```pip install -e "C:/ruta/a/surpRISe"```

**Nota:** La carpeta dist/ está en el .gitignore. No se sube a GitHub; se genera localmente en cada PC cuando se necesita.

## 5. Finalizar la Migración (Merge a Main)

### Pasos para fusionar la rama de trabajo con la rama principal de forma segura:

1. Asegurar que todos los cambios están en el commit:
   ```bash
   git add .
   git commit -m "Finalize uv migration and refactor"
   ```

2. Cambiar a la rama principal (main):
   ```bash
   git checkout main
   ```

3. Asegurar que main está actualizada:
   ```bash
   git pull origin main
   ```

4. Fusionar la rama de migración en main:
   ```bash
   git merge uv-migration
   ```

5. Subir los cambios definitivos a GitHub:
   ```bash
   git push origin main
   ```

6. (Opcional) Borrar la rama de migración localmente:
   ```bash
   git branch -d uv-migration
   ```