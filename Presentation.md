# Überblick
+ Datenstruktur
+ Fragestellungen

# Kohorten

Heatmap: 
Abszisse: Alter, Ordinate: Geschlecht, Choropleten: Überlebensrate

Fragestellung: Wie sieht es mit dem Überleben ganzer Familien aus? Wie wurden diese auseinandergerissen?

Frauen und Kinder zuerst: Maximierung der geretteten Lebenszeit.

Welche Informationen fehlen? Welche Geschichten gingen dadurch verloren?

# Decision Tree

Modellierung: Wie kann ich meine Überlebenschance vergrössern, wenn ich auf der Titanic einschiffe?

### Allegmein
Entscheidungsbäume sind nicht-parametrische, überwachte Lernverfahren für Klassifikation und Regression. Sie approximieren die Zielvariable durch eine Folge hierarchischer if-then-else-Regeln (stückweise konstante Approximation).

Ein Entscheidungsbaum funktioniert im Grunde wie das Spiel **„20 Fragen“** oder ein medizinisches Flussdiagramm: Er stellt nacheinander Ja/Nein-Fragen, um eine Gruppe von Menschen schrittweise in immer eindeutigere Untergruppen zu sortieren.

### 1. Die wirksamste Frage zuerst

Zu Beginn stehen alle Passagiere in einem einzigen Topf. Der Baum sucht nun nach der einen Eigenschaft, die Überlebende und Verstorbene am saubersten trennt.

Dafür testet er alle Merkmale durch:

- Alter?
    
- Ticketklasse?
    
- Geschlecht?
    

Bei der Titanic war das Geschlecht der stärkste Faktor („Frauen und Kinder zuerst“). Fragt man zuerst: _„Ist die Person eine Frau?“_, hat man sofort zwei deutlich klarere Gruppen als vorher:

- **Gruppe A (Frauen):** Hier haben rund 75 % überlebt.
    
- **Gruppe B (Männer):** Hier haben nur rund 19 % überlebt.
    

### 2. Weitersortieren in den Untergruppen

Nun wiederholt der Baum das gleiche Prinzip getrennt für beide Seiten:

- **Bei den Frauen:** Die meisten haben überlebt, aber nicht alle. Der Baum fragt als Nächstes nach der **Reiseklasse**: Frauen in der 1. und 2. Klasse überlebten fast ausnahmslos; in der 3. Klasse starben deutlich mehr.
    
- **Bei den Männern:** Die meisten sind verstorben. Der Baum sucht nach Ausnahmen und fragt nach dem **Alter**: Jungen unter 14 Jahren hatten noch eine solide Rettungschance, erwachsene Männer kaum.
    

### 3. Das Endergebnis (Die Blätter)

Nach zwei bis drei Fragen teilt sich der Baum nicht mehr weiter auf. Am Ende jedes Pfades steht ein „Blatt“ – das ist eine feste Kohorte.

In jedem Blatt zählt der Baum schlicht nach, was damals historisch passiert ist:

- **Pfad:** _Frau → 1. Klasse_ $\rightarrow$ 97 % Überlebensrate.
    
- **Pfad:** _Mann → über 14 Jahre alt → 3. Klasse_ $\rightarrow$ ca. 12 % Überlebensrate.
    

Kommt nun ein neuer Passagier hinzu, läuft er den passenden Pfad entlang. Seine Überlebenschance entspricht dem Prozentsatz der Gruppe, in deren Blatt er landet.
#  Seenotfälle

Welche Risiken gibt es in der Seefahrt? Was wird getan, um Katastrophen wie die Titanic zu verhindern? 
+ Funkstation 24h besetzen
+ Keine Warnungen ignorieren
+ Ehrenkodex
+ Rote Signalraketen verbindlich
+ Einheitliche Notsignalisation .......-

Was kann man nicht vermeiden: 
Ein Schiff ab einer Länge von 25m legt man Ruder und es tut sich erstmal: gar nichts.
Schnelles Ausweichen geht nicht, Kavitation der Schraube bei Umkehrschub verhindert wirksame Steuerung (Ruder wird nicht mehr angeströmt).

