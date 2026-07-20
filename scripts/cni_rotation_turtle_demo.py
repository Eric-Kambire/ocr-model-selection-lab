"""Animation Turtle autonome pour comprendre la rotation d'une CNI.

Ce script est pédagogique : il n'ouvre pas de PDF et ne remplace pas le
laboratoire de crop. Il visualise, dans une fenêtre distincte, les deux phases
de recherche d'angle utilisées par la méthode Pillow.

Lancement :
    python scripts/cni_rotation_turtle_demo.py --best-angle -48

Touches : espace = pause/reprendre ; r = recommencer ; Échap = fermer.
"""

from __future__ import annotations

import argparse
import math
import turtle
from dataclasses import dataclass


# Ratio physique de référence d'une carte ISO ID-1 : largeur / hauteur.
TARGET_RATIO = 1.586


@dataclass(frozen=True)
class Frame:
    """Une image de l'animation, avec sa phase et son angle candidat."""

    phase: str
    angle: float


def build_frames(best_angle: float) -> list[Frame]:
    """Construit la recherche large, puis l'affinage autour du meilleur angle."""
    # La recherche réelle Pillow utilise un pas de 3 degrés.
    coarse = [Frame("Recherche large", float(angle)) for angle in range(-90, 91, 3)]
    # L'affinage réel examine chaque degré autour du meilleur candidat large.
    rounded_best = round(best_angle / 3.0) * 3.0
    fine = [Frame("Affinage", float(angle)) for angle in range(int(rounded_best) - 3, int(rounded_best) + 4)]
    # La dernière image montre la transformation finalement appliquée.
    return coarse + fine + [Frame("Rotation finale", float(best_angle))]


def expected_ratio(candidate_angle: float, best_angle: float) -> float:
    """Produit un score illustratif, pas une mesure prise sur une vraie image.

    Dans la vraie pipeline, le ratio est obtenu du rectangle englobant du
    masque binaire. Ici, cette fonction permet seulement de rendre la recherche
    lisible sans nécessiter un PDF dans le script Turtle.
    """
    return TARGET_RATIO + abs(candidate_angle - best_angle) * 0.012


class RotationDemo:
    """Dessine l'état courant et avance automatiquement image par image."""

    def __init__(self, best_angle: float, delay_ms: int) -> None:
        self.best_angle = best_angle
        self.delay_ms = delay_ms
        self.frames = build_frames(best_angle)
        self.index = 0
        self.playing = True

        # La fenêtre conserve une palette neutre, comme le laboratoire de crop.
        self.screen = turtle.Screen()
        self.screen.title("CNI — recherche et affinage de rotation")
        self.screen.setup(width=1100, height=720)
        self.screen.bgcolor("#edf0f3")
        self.screen.tracer(0, 0)

        # Un seul crayon est réutilisé : aucune trace de l'image précédente.
        self.pen = turtle.Turtle(visible=False)
        self.pen.speed(0)
        self.pen.penup()

        # Les interactions permettent de ralentir ou rejouer l'explication.
        self.screen.onkey(self.toggle_play, "space")
        self.screen.onkey(self.restart, "r")
        self.screen.onkey(self.screen.bye, "Escape")
        self.screen.listen()

    def toggle_play(self) -> None:
        """Met en pause ou reprend l'animation sans remettre les calculs à zéro."""
        self.playing = not self.playing
        if self.playing:
            self.tick()

    def restart(self) -> None:
        """Revient au premier angle de la recherche large."""
        self.index = 0
        self.playing = True
        self.tick()

    def rotated_rectangle(self, width: float, height: float, angle: float, *, color: str, fill: str | None = None) -> None:
        """Dessine un rectangle centré en (0, 0) et incliné selon ``angle``."""
        self.pen.color(color)
        # On se déplace du centre au coin haut-gauche dans le repère tourné.
        self.pen.goto(0, 0)
        self.pen.setheading(angle)
        self.pen.backward(width / 2)
        self.pen.left(90)
        self.pen.backward(height / 2)
        self.pen.right(90)
        if fill:
            self.pen.fillcolor(fill)
            self.pen.begin_fill()
        self.pen.pendown()
        for _ in range(2):
            self.pen.forward(width)
            self.pen.left(90)
            self.pen.forward(height)
            self.pen.left(90)
        if fill:
            self.pen.end_fill()
        self.pen.penup()

    def write(self, text: str, x: float, y: float, *, size: int = 13, color: str = "#173a72", align: str = "left") -> None:
        """Écrit une ligne explicative dans la fenêtre Turtle."""
        self.pen.color(color)
        self.pen.goto(x, y)
        self.pen.write(text, align=align, font=("Arial", size, "normal"))

    def draw(self, frame: Frame) -> None:
        """Dessine la CNI, le canevas agrandi et les calculs du candidat courant."""
        self.pen.clear()
        theta = math.radians(frame.angle)
        cosine, sine = math.cos(theta), math.sin(theta)
        card_width, card_height = 320.0, 202.0
        canvas_width = abs(card_width * cosine) + abs(card_height * sine)
        canvas_height = abs(card_width * sine) + abs(card_height * cosine)
        measured_ratio = expected_ratio(frame.angle, self.best_angle)
        score = abs(measured_ratio - TARGET_RATIO)

        # Titre et état de la recherche.
        self.write("Rotation d'une CNI : recherche puis affinage", 0, 320, size=22, align="center")
        self.write(f"Phase : {frame.phase}", -510, 275, size=16)
        self.write(f"Candidat : θ = {frame.angle:+.0f}° = {theta:+.4f} rad", -510, 245, size=15)

        # Le rectangle pointillé représente le nouveau canevas créé avec expand=True.
        self.pen.goto(-canvas_width / 2, -canvas_height / 2)
        self.pen.color("#8993a4")
        self.pen.pensize(1)
        self.pen.pendown()
        for _ in range(2):
            self.pen.forward(canvas_width)
            self.pen.left(90)
            self.pen.forward(canvas_height)
            self.pen.left(90)
        self.pen.penup()

        # La carte est tournée autour de son centre : le centre reste fixe.
        self.rotated_rectangle(card_width, card_height, frame.angle, color="#1769d1", fill="#dcebcf")
        self.pen.goto(0, 0)
        self.pen.dot(8, "#b42318")
        self.write("centre de rotation", 12, 8, size=12, color="#6e1d16")

        # Le panneau de droite expose les mêmes formules que le code applique.
        self.write("Calculs", 280, 210, size=18)
        self.write(f"cos(θ) = {cosine:+.4f}", 280, 175)
        self.write(f"sin(θ) = {sine:+.4f}", 280, 150)
        self.write("x' = cosθ(x−cx) − sinθ(y−cy) + cx", 280, 105, size=12)
        self.write("y' = sinθ(x−cx) + cosθ(y−cy) + cy", 280, 80, size=12)
        self.write(f"Canevas : {canvas_width:.0f} × {canvas_height:.0f} px", 280, 35)
        self.write("(il peut grandir : les coins ne sont pas coupés)", 280, 10, size=11, color="#58677d")

        # Important : le ratio affiché est un exemple de score, pas un OCR.
        self.write(f"Ratio illustratif : {measured_ratio:.3f}", -510, -255, size=15)
        self.write(f"Score = |ratio − 1,586| = {score:.3f}", -510, -280, size=15)
        self.write("La vraie application calcule ce ratio sur le contour du masque binaire.", -510, -315, size=11, color="#58677d")
        self.write("Espace : pause/reprendre   ·   R : rejouer   ·   Échap : fermer", 0, -345, size=12, align="center", color="#58677d")
        self.screen.update()

    def tick(self) -> None:
        """Affiche la frame suivante ; Turtle programme la suivante sans bloquer la fenêtre."""
        if not self.playing:
            return
        if self.index >= len(self.frames):
            self.playing = False
            return
        self.draw(self.frames[self.index])
        self.index += 1
        self.screen.ontimer(self.tick, self.delay_ms)

    def run(self) -> None:
        """Démarre l'animation et laisse Turtle gérer la boucle graphique."""
        self.tick()
        self.screen.mainloop()


def main() -> None:
    """Lit les paramètres de démonstration et ouvre la fenêtre Turtle."""
    parser = argparse.ArgumentParser(description="Animation pédagogique de la rotation CNI.")
    parser.add_argument("--best-angle", type=float, default=-48.0, help="Angle final simulé, entre -90 et +90 degrés.")
    parser.add_argument("--delay-ms", type=int, default=90, help="Durée entre deux images de l'animation.")
    args = parser.parse_args()
    if not -90.0 <= args.best_angle <= 90.0:
        parser.error("--best-angle doit être compris entre -90 et +90.")
    if args.delay_ms < 20:
        parser.error("--delay-ms doit être au minimum 20.")
    RotationDemo(args.best_angle, args.delay_ms).run()


if __name__ == "__main__":
    main()
