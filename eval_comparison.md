# Student before/after distillation — held-out comparison

Held-out size: **8**

## Aggregate metrics

| metric | before | after | Δ |
|---|---|---|---|
| label-in-answer rate | 0.375 | 0.500 | +0.125 |
| teacher-text similarity | 0.183 | 0.318 | +0.135 |
| gold-label similarity | 0.077 | 0.025 | -0.052 |

## Per-sample (sorted by Δ teacher-similarity)

### qa_id=3909  (Temporal)
- **Q:** What action do the players perform before bent leg jump?
- **Gold:** `aerobic gymnastics split`
- **Δ tsim:** +0.334  label match: 0→0

  **Teacher (27B) gen:**
  > The clip shows three female aerobic gymnasts performing on a marked court. At approximately 0:00–0:01, all three execute a simultaneous forward roll (also known as a somersault) starting from a standing position, tucking their bodies and rolling forward along the floor. Immediately after completing this roll, they rise into a bent-leg jump (a two-footed leap with knees drawn up toward the chest).

So, the action performed immediately before the bent-leg jump is a **forward roll**.

  **Student BEFORE:**
  > The players perform a back handspring before jumping into a bent-leg jump.

  **Student AFTER:**
  > The clip shows four female aerobics gymnasts performing on a competition mat. At approximately the 0:00 mark, the group executes a synchronized movement where they leap into the air while simultaneously extending their legs straight out to the sides in a horizontal position. This specific skill is known as a straddle jump or simply a straddle leap. Immediately following this jump, at roughly 0:01, they land and transition directly into the "bent leg jump" mentioned in the prompt.

Therefore, the action performed immediately before the bent leg jump is a **straddle jump**.

---

### qa_id=15927  (Causal)
- **Q:** What action does the shoot of the right team cause?
- **Gold:** `football block`
- **Δ tsim:** +0.288  label match: 0→0

  **Teacher (27B) gen:**
  > The clip shows a Premier League match between Brighton (BHA) and Manchester United (MU), with MU leading 3–0 at 59:13. The red-clad players are Manchester United; they’re taking a free kick from just outside the penalty area. At ~0.2s, a MU player strikes the ball low and hard toward the left side of the goal. The ball travels quickly past the defensive wall and under the leap of the Brighton goalkeeper, who dives but cannot reach it. By 0.3s, the ball has crossed the goal line into the net — clearly visible as it hits the back of the net behind the keeper.

This is a direct free-kick goal. Th

  **Student BEFORE:**
  > The shot causes the right team to make a tackle on the left team.

  **Student AFTER:**
  > The clip shows a set-piece situation in a Premier League match between Brighton & Hove Albion (BHA) and Manchester United (MU). At approximately 0:02–0:03, a player from the right-side attacking team (Manchester United, in red) takes a free kick or corner from just outside the penalty area. The ball arcs toward the goal. A Brighton defender (in blue/white) attempts to block but fails to stop the ball. The ball crosses the goal line and enters the net — we can clearly see it cross the line and hit the back of the net before any rebound or deflection occurs.

This is a direct scoring opportunity

---

### qa_id=78750  (Temporal)
- **Q:** What do the players perform before performing salto forward tucked to cross stand?
- **Gold:** `salto sideward tucked, take off from one leg to side stand`
- **Δ tsim:** +0.231  label match: 0→0

  **Teacher (27B) gen:**
  > In the clip at approximately 0:01–0:02, we see a female gymnast on the balance beam preparing for her routine. She begins standing upright with arms raised, then executes a controlled backward movement into a handstand position before transitioning into a forward tuck salto that lands back into a standing position on the beam.

The key sequence is:

1. **Starting Position**: Standing tall with arms extended overhead.
2. **Transition Move**: She performs a **back handspring** (also called a back walkover variation) — this involves arching backward into a handstand while keeping legs together, t

  **Student BEFORE:**
  > The players first perform a jump on the balance beam.

  **Student AFTER:**
  > The athlete performs a back handspring into a layout position, followed by a front aerial, before transitioning into the salto forward tucked to cross stand. The sequence begins at approximately 0:01 when she initiates the back handspring, moves into a layout position mid-air, lands on her feet facing forward, and then executes the salto forward tucked to cross stand. This combination demonstrates advanced skills typical in artistic gymnastics routines on the balance beam.

---

### qa_id=9623  (Counterfactual)
- **Q:** Would the right team succeed in doing spike if the other team did not save the ball?
- **Gold:** `yes`
- **Δ tsim:** +0.082  label match: 0→1

  **Teacher (27B) gen:**
  > Yes. At roughly 0:04–0:05, Turkey’s middle blocker (#8 Bahar Toksoy) executes a quick hit from the center position. The ball leaves her hand on a sharp downward trajectory toward Brazil’s front-right zone near the net. A Brazilian defender dives and makes contact with the ball at about 0:06, keeping it alive; without that dig, the ball would have landed inbounds on Brazil’s side. The spike’s angle, speed, and placement indicate it was going to score had the block/dig not occurred.

  **Student BEFORE:**
  > No, because the opponent saved the spike and missed it. The correct answer is that the left team did not succeed in doing a spike.

  **Student AFTER:**
  > Yes.

At roughly 0:04–0:05, the yellow team on the right executes a successful spike that lands inbounds on the opponent’s side. The black player on the left attempts a dig but fails to keep the ball alive; the ball hits the floor cleanly inside the boundary lines. There is no visual indication of a block deflection or net touch altering the trajectory enough to prevent the kill. Therefore, absent any defensive attempt, the spike results directly in a point for the right team.

---

### qa_id=42695  (Descriptive)
- **Q:** Do the players perform handspring forward on, tucked salto forward with 0.5 turn off?
- **Gold:** `no`
- **Δ tsim:** +0.079  label match: 1→1

  **Teacher (27B) gen:**
  > No.

What I see: A single female gymnast performs a tumbling pass on the springboard runway into a blue mat. She executes a round-off onto the board, followed by a back handspring (or similar backward skill) that launches her into a double tuck somersault. The entire sequence is backward-oriented; there is no forward-facing salto or any half-turn variation.

Evidence: At ~0:06–0:08 she lands facing away from the direction of travel after the second backward rotation, consistent with a double back tuck. There is no moment where she rotates forward or completes a 0.5 turn off any element.

Verdi

  **Student BEFORE:**
  > No. The players execute a handspring forward, but they do not perform a tuck salto forward with a 0.5-degree turn off. Instead, they complete a routine that includes a handspring forward, followed by a back handspring, and then a front handspring.

  **Student AFTER:**
  > No.

What I see in the clip:
- Sport: Artistic gymnastics, women’s vault.
- The athlete performs a round-off onto the springboard, followed by a front handspring entry into a vault table.
- From the table she executes a twisting somersault (a Yurchenko-style vault with multiple twists) and lands facing forward.

Technical breakdown:
- There is no "handspring forward" performed on the apparatus; that phrasing doesn't match any standard vault skill.
- The flight phase is a twisting somersault, not a tucked forward salto with a half-turn.
- The rotation appears to be more than 180 degrees (likely

---

### qa_id=10212  (Causal)
- **Q:** Why is the spike of the right team failed?
- **Gold:** `volleyball saved`
- **Δ tsim:** +0.062  label match: 0→0

  **Teacher (27B) gen:**
  > At approximately 0:01 in the clip, we observe a player from the red team (right side) executing a spike attempt during a CEV Tokyo Volleyball Qualification match. The setter has set the ball near the net on the right side, and the attacker leaps to strike it.

However, just as the spiker makes contact, a defender from the opposing black-and-white team (left side) jumps at the net and successfully blocks the ball. The block is well-timed and positioned directly in front of the spiker’s hitting path. As a result, the ball does not cross into the opponent’s court — instead, it is deflected upward

  **Student BEFORE:**
  > The spike of the right team failed because it landed on the net. This indicates that the player attempting to hit the spike missed the target area. The visual evidence shows the spike hitting the net directly, which signifies a miss during the spike attempt.

  **Student AFTER:**
  > The clip shows a women’s indoor volleyball match (CEV Tokyo Volleyball Qualification 2020) between Turkey (TUR, red/blue) on the left side of the net and Germany (GER, white/red) on the right. At roughly 0:01–0:02, Germany sets up an attack near the net; their middle hitter attempts a quick spike but is met at the net by two Turkish blockers who time their jump perfectly and form a tight block. The ball deflects sharply upward and back toward Germany’s side, where a German defender immediately moves to dig it. Because the spike lacked enough downward trajectory and penetration into the block z

---

### qa_id=25149  (Counterfactual)
- **Q:** Would the cross of the right team be successful if the other team did not intercept?
- **Gold:** `yes`
- **Δ tsim:** +0.020  label match: 1→1

  **Teacher (27B) gen:**
  > Yes. At roughly 66:04, the white-and-black striped player on the right wing delivers a low, driven cross into the six-yard box. The trajectory is aimed directly at the near post where a teammate is already positioned for a header. There is no defender between the ball’s path and that attacker; the only reason the play fails is because a red-shirted defender steps in and clears the ball mid-flight. Without that interception, the ball would have arrived at the striker’s head unimpeded, making a goal highly probable—likely a tap-in or close-range finish. So, absent the defensive intervention, the

  **Student BEFORE:**
  > Yes, the cross of the right team would likely be successful. The video shows the left team attempting a shot on goal, but the right team successfully intercepted the ball before it could hit the backboard. This interception prevented the left team from scoring, indicating that their defensive response was effective against the opponent's attack. Therefore, without the right team's intervention, the shot would have resulted in a potential goal for the left team.

  **Student AFTER:**
  > The clip shows a soccer match at roughly 66:04. A player on the right side (Leeds United) delivers a low cross into the penalty area toward his teammate making a run near the six-yard box. The ball arrives just inside the box; a defender from the opposing team (Huddersfield Town) steps up and clears it with a tackle.

If the defender had not intercepted, the cross would have reached the intended attacker who was already sprinting into space behind the defensive line. Given the timing and placement, the cross was well-timed to beat the defense and create a clear scoring opportunity. Therefore, 

---

### qa_id=13664  (Descriptive)
- **Q:** Does any team gain a point?
- **Gold:** `no`
- **Δ tsim:** -0.020  label match: 1→1

  **Teacher (27B) gen:**
  > No. The clip shows USA attacking and taking a long-range shot that goes wide; the ball does not enter the net. No goal is scored, so neither team gains a point in this sequence.

  **Student BEFORE:**
  > No. The scoreboard shows USA 5 - 2 JPN. No points were scored in this clip.

  **Student AFTER:**
  > No team gains a point in this clip.

At roughly 0:00–0:01, we see a USA player (white jersey) dribbling downfield on the right flank while being closed down by Japanese defenders (blue jerseys). The camera follows the ball upfield; there’s no shot taken, no goal scored, and no set piece that results in a goal. The scoreboard remains unchanged at 5–2 until the clip ends. Since the ball never enters the net and no scoring event occurs, the answer is definitively no.

---

