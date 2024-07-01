### TFG: Ús de models de difusió per alterar la realitat física de les escenes

**Autora:** Abril Piñol Chacon - 1604159
**Tutor:** Ramon Baldrich Caselles
Curs 2023-24, Universitat Autònoma de Barcelona (UAB)


Aquest és el dossier del projecte de final de grau d'Enginyeria de Dades. A continuació és mostra l'abstract, juntament amb informació del contingut.


## Abstract
Aquest article explora la capacitat dels models Stable Diffusion de controlar la il·luminació en les imatges generades. Tot i ser un camp en creixement i haver-hi avenços recents, és una tasca on es continuen trobant  moltes inconsistències. Aquesta recerca pretén trobar la manera de controlar la il·luminació utilitzant inputs tan senzills com sigui possible. Mitjançant una sèrie d'experiments i una anàlisi profunda dels resultats, la recerca conclou que mentre els models SD mostren un bon rendiment, el factor crític per a tasques complexes rau en les xarxes auxiliars que guien amb precisió la sortida. Aquest treball és exploratori i mostra com els models SD poden adaptar-se a diverses tasques de complexitat variable i la influència en la qualitat de la sortida del model d'alguns dels paràmetres o de l'ús de pesos preentrenats.

**Paraules clau:** Relighting, Stable Diffusion, condicionar, dataset, imatges reals, imatges sintètiques, image restoration, light probe.


## Estructura 

Aquest repositori conté els següents elements principals:

```
├───Repositoris
    ├───ControlNet-abril
    ├───img2img-turbo-abril
    ├───ir-sde-abril
    └───papers
├──Documents
    ├───Informe Inicial
    ├───Informe de Progrés I
    ├───Informe de Progrés II
    ├───Proposta d'Informe Final
    ├───Informe Final
    └───zip Informe Final LaTeX
```

### Descripció dels Elements

1. **Repositoris/**: Aquesta carpeta conté els repositoris clonats que s'utilitzen per als experiments del projecte. Cada subcarpeta dins de "Repositoris" correspon a un repositori específic proposat per altres autors, amb petites variacions realitzades per adaptar-los al context i requisits del projecte.

2. **Repositoris/papers/**: En aquesta subcarpeta es troben els papers que proposen els codis esmentats anteriorment.

3. **Documents/**: Aquí s'emmagatzemen els diferents informes generats al llarg del projecte, que documenten el progrés i el treball realitzat:
   - **Informe Inicial**
   - **Informe de Progrés I**
   - **Informe de Progrés II**
   - **Proposta d'Informe Final**
   - **Informe Final**
   - **zip Informe Final LaTeX**

Aquest repositori està organitzat de manera que facilita la gestió i seguiment del projecte, mantenint tots els recursos necessaris accessibles i ben estructurats.