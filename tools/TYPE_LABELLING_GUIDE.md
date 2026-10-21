# Guía de etiquetado — Tipo de vehículo

Objetivo: que dos anotadores (o vos en dos días distintos) miren el mismo
vehículo y elijan **la misma clase**. La meta no es "acertar", es ser
**consistente**. Un dataset consistente vale más que uno perfecto pero ruidoso.

Se etiqueta observando la **forma completa de la carrocería**, en la vista que
la cámara de la puerta muestra (mayormente trasera al ingresar, frontal al
salir). Si el vehículo está cortado, muy oscuro o borroso, **descartá** la
muestra (`d`): no la fuerces a una clase.

---

## Las 7 clases

| clase | qué es | pista decisiva |
|---|---|---|
| **motocicleta** | moto | dos ruedas |
| **automovil** | sedán | 3 volúmenes: capó + cabina + **baúl saliente**; techo bajo |
| **vagoneta** | SUV / hatchback / rural / van ligera | techo alto que **llega hasta el portón trasero**; sin baúl saliente |
| **camioneta** | pickup | **caja de carga abierta** atrás |
| **furgon** | van de carga | van **sin ventanas** laterales traseras; cerrada |
| **minibus** | van/minibús de pasajeros | van **con ventanas** y muchas plazas |
| **otro** | bus, microbús grande, camión | vehículo **pesado/comercial** grande |

---

## Árbol de decisión (en este orden)

1. ¿Dos ruedas? → **motocicleta**
2. ¿Tiene **caja de carga abierta** atrás? → **camioneta** (aunque sea doble cabina)
3. ¿Es un **camión o bus grande** (chasis pesado, muchas ruedas, gran volumen)? → **otro**
4. ¿Es una **van**?
   * sin ventanas laterales traseras → **furgon**
   * con ventanas y muchas plazas → **minibus**
5. ¿Es un **auto chico**?
   * tiene **baúl saliente** y techo bajo (3 volúmenes) → **automovil**
   * el techo **baja hasta el portón**, sin baúl (2 volúmenes / SUV) → **vagoneta**

Si dudás entre dos, **aplicá el orden de arriba hacia abajo**: gana el primero
que cumpla. Así todos resuelven la duda igual.

---

## Las 3 confusiones reales (y cómo cortarlas)

### 1. automovil ↔ vagoneta  ← la que más te está costando
* **automovil**: hay una **"cola" con baúl"** que sobresale y el techo es más bajo en la parte trasera.
* **vagoneta**: el techo es **continuo hasta el portón**; parece un "cubo" con ventana trasera grande.
* Regla corta: **¿hay baúl que sobresale? automovil. ¿No? vagoneta.**
* Ojo: un hatchback chico NO es automovil → es **vagoneta**.

### 2. vagoneta ↔ minibus
* **vagoneta**: SUV/van chica, uso particular (3-5 plazas).
* **minibus**: van de **pasajeros** con fila de ventanas y muchas plazas (transporte).
* Regla corta: **muchas ventanas + tamaño van + transporte → minibus.**

### 3. furgon ↔ minibus
* **furgon**: **sin ventanas** traseras (transporte de carga).
* **minibus**: **con ventanas**.
* Regla corta: **¿se ven ventanas atrás? minibus. ¿Pared ciega? furgon.**

---

## Reglas de oro

1. Etiquetá por la **silueta completa**, nunca por una parte o por la marca.
2. Una muestra = un vehículo principal. Si hay dos vehículos, etiquetá el que
   domina la imagen.
3. No infieras por el color ni por el modelo ("es un Toyota" no decide la clase).
4. **Consistencia > criterio personal.** Ante la duda, seguí el árbol y el orden.
5. Si no se distingue con seguridad → **descartar** (`d`), no adivinar.
6. `otro` NO es "no sé". Es exclusivamente bus / microbús grande / camión.
   Un vehículo que no sabés clasificar se descarta, no va a `otro`.
