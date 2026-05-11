# Flip-rate report — distillation before vs. after
Threshold (best_window_sim ≥): **0.8**  ·  n = **1000** held-out samples
## Buckets
| bucket | n | share |
|---|---:|---:|
| **wrong → correct (FIX)** | 174 | 17.4% |
| correct → wrong (REGRESS) | 36 | 3.6% |
| correct → correct         | 191 | 19.1% |
| wrong → wrong             | 599 | 59.9% |

## Aggregate similarity
| metric | before | after | Δ |
|---|---:|---:|---:|
| correct rate (gold-fuzzy ≥ 0.8) | 0.227 | 0.365 | +0.138 |
| avg gold best-window sim | 0.546 | 0.712 | +0.166 |
| avg teacher-text sim     | 0.000 | 0.000 | +0.000 |

## By sport
| sport | n | fix | regress | persist_ok | persist_bad | correct_after |
|---|---:|---:|---:|---:|---:|---:|
| aerobic_gymnastics | 200 | 44 | 5 | 21 | 130 | 0.325 |
| basketball | 200 | 14 | 19 | 60 | 107 | 0.370 |
| fg | 200 | 40 | 6 | 37 | 117 | 0.385 |
| football | 200 | 33 | 3 | 33 | 131 | 0.330 |
| volleyball | 200 | 43 | 3 | 40 | 114 | 0.415 |

## By question_type
| qtype | n | fix | regress | persist_ok | persist_bad | correct_after |
|---|---:|---:|---:|---:|---:|---:|
| Causal | 153 | 1 | 0 | 0 | 152 | 0.007 |
| Counterfactual | 150 | 46 | 8 | 62 | 34 | 0.720 |
| Descriptive | 391 | 115 | 25 | 111 | 140 | 0.578 |
| Temporal | 306 | 12 | 3 | 18 | 273 | 0.098 |

## Flip-to-correct rows  (174)
Sorted by Δ gold-similarity (largest gain first).

### qa_id=28812  ·  aerobic_gymnastics / Descriptive
- **Q:** How many players are performing gymnastics?
- **Gold:** `2`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** Two players are performing gymnastics.

  **AFTER:**  Two athletes are performing on the floor during this aerobic gymnastics routine. They execute synchronized movements including jumps and turns, which are characteristic of group aerobic routines. No additional performers enter or leave the frame during these frames.

Answer: 2

---

### qa_id=28861  ·  aerobic_gymnastics / Descriptive
- **Q:** How many actions do the players perform?
- **Gold:** `2`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The players perform two distinct actions: they jump and jump again. The first jump occurs at approximately 0.0 seconds, and the second jump happens at roughly 0.4 seconds.

  **AFTER:**  In this rhythmic gymnastics group routine, we observe three distinct movements performed by all three athletes:

1. **Split leap (or straddle jump)** – All three athletes execute a leap where their legs extend into a full split position mid-air. This is clearly seen at approximately 0:01–0:02 in the clip.

2. **Round-off** – Immediately following the split leap, each athlete lands on one foot and transitions into a round-off (a cartwheel-like motion with hands planted firmly). This occurs around 0:02–0:03.

3. **Breakaway** – After completing the round-off, they launch into a breakaway (a dyna

---

### qa_id=28863  ·  aerobic_gymnastics / Descriptive
- **Q:** How many times do the players perform explosive support?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The players perform one explosive support. This occurs at approximately 0.6 seconds when they jump up and down on their feet.

  **AFTER:**  In this rhythmic gymnastics group routine (France, Bugalho–Le Mesle–Orta), we observe three athletes performing synchronized movements on the floor. The key element in question is “explosive support,” which typically refers to a dynamic movement where one athlete supports another horizontally while airborne or transitioning mid-air — often involving a leap or jump with legs extended in a horizontal plane.

Looking at the clip:

- At approximately 0:00–0:01, all three athletes simultaneously execute a leap where they extend their bodies horizontally, supporting each other mid-air. This is clear

---

### qa_id=28864  ·  aerobic_gymnastics / Descriptive
- **Q:** How many players perform bent leg jump?
- **Gold:** `3`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** One player performs bent leg jump.

  **AFTER:**  In the provided clip, we observe three female rhythmic gymnasts performing on a floor exercise mat during a competition. At approximately 0:01–0:02, all three simultaneously execute a leap where their legs are brought toward their chest while airborne — this is technically known as a "bent-leg jump" (or sometimes called a “tuck jump” depending on exact form, but here they appear fully extended at the top of the jump).

All three athletes clearly perform the movement together in unison. There is no indication that any athlete skips or modifies the move; each completes the full motion cleanly.



---

### qa_id=27784  ·  aerobic_gymnastics / Descriptive
- **Q:** How many times do the players perform explosive push up?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The players perform explosive push ups once during this clip.

  **AFTER:**  In this aerobic gymnastics routine, we observe two distinct moments where the athletes execute an explosive push-up motion.

1.  **First Push-Up:** At approximately the 0.6-second mark, all four athletes simultaneously drop into a plank position and push their bodies upward explosively, transitioning directly into a handstand-like pose. This is one clear instance of the explosive movement.

2.  **Second Push-Up:** Around the 1.4-second mark, the group moves again. Two athletes (one on the left side and one in the center) perform another synchronized explosive push-up, propelling themselves bac

---

### qa_id=28470  ·  aerobic_gymnastics / Descriptive
- **Q:** How many players are performing gymnastics?
- **Gold:** `2`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** Two.

  **AFTER:**  In the provided clip, two female athletes are visible on the floor performing synchronized gymnastics moves. Both are actively engaged in their routine, executing leaps and turns in unison. No other performers enter or leave the frame during this segment.

Answer: **2**

---

### qa_id=28064  ·  aerobic_gymnastics / Descriptive
- **Q:** How many players perform straddle jump?
- **Gold:** `3`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** Three players perform straddle jump.

  **AFTER:**  In the provided clip from the FIG World Cup Aerobic Gymnastics competition, we observe three female athletes performing synchronized movements on the floor. At approximately 0:01–0:02 into the clip, all three athletes simultaneously execute a straddle jump — where they leap into the air with legs spread wide apart (one forward, one back) while keeping their bodies straight and arms extended overhead.

This is clearly visible as each athlete lifts fully into the straddle position mid-air before landing. The movement is identical across all performers, indicating it is part of their choreographe

---

### qa_id=28943  ·  aerobic_gymnastics / Descriptive
- **Q:** How many times do the players perform support?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The players perform support once.

  **AFTER:**  In this rhythmic group gymnastics routine, we observe two distinct moments where athletes execute a "support" element: a handstand-like position where one leg is extended forward and the other back, while the torso leans on the hands.

1.  **First Support:** At approximately the 0:01 mark, the female athlete on the far left performs a full handstand support. She balances entirely on her hands, showcasing strength and control.
2.  **Second Support:** Immediately following that, at roughly 0:02, another athlete (center-left) also executes a support. This time, she is performing a more complex va

---

### qa_id=28925  ·  aerobic_gymnastics / Descriptive
- **Q:** How many types of actions do the players perform?
- **Gold:** `2`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The players are performing two distinct types of actions: jumping on hands and knees (or similar low-impact movements) and sitting down on their heels while holding onto a wall.

  **AFTER:**  In the provided clip, we observe two distinct phases of action performed by the athletes:

1. **Initial Positioning**: At the start of the clip (around 0.0–0.4 seconds), both athletes are seen lying flat on their stomachs on the mat. This is the first type of action — a prone position.

2. **Transition to Lunge/Sit**: Immediately following the initial pose, they rise and transition into a lunge-like position (around 0.5–0.8 seconds), where one leg extends forward and the other back, while the torso leans forward. This is the second type of action — a dynamic movement that involves shifting wei

---

### qa_id=29291  ·  aerobic_gymnastics / Descriptive
- **Q:** How many players perform pike jump?
- **Gold:** `5`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** 4

  **AFTER:**  In the provided clip, at approximately the 0.8–1.0 second mark, we observe five female athletes performing on the aerobic gymnastics mat. They execute synchronized movements that include leaps into the air with legs extended straight out to their sides while keeping their bodies straight and relatively flat — this is characteristic of a **pike jump**.

All five athletes complete the pike jump simultaneously during this sequence. There is no indication in the footage that any athlete fails to perform it; all five clearly land cleanly and continue into the next phase of their routine.

✅ Final A

---

### qa_id=28075  ·  aerobic_gymnastics / Descriptive
- **Q:** How many types of actions do the players perform?
- **Gold:** `2`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The players in the clip are performing two distinct types of actions: jumping and spinning. One player jumps while another spins.

  **AFTER:**  In this aerobic gymnastics routine, we observe four distinct types of actions performed by the athletes:

1. **Running steps**: The athletes move forward while keeping their legs together and upright, maintaining rhythm and form. This is seen throughout the sequence as they transition into more dynamic movements.

2. **Split leaps**: One athlete performs a full split leap mid-air, showcasing flexibility and control. Another athlete executes a split on the ground during a pose.

3. **Push-ups**: Several athletes perform push-ups as part of their choreography, demonstrating core strength and fle

---

### qa_id=28768  ·  aerobic_gymnastics / Descriptive
- **Q:** How many players perform explosive push up?
- **Gold:** `3`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** Three players perform explosive push ups.

  **AFTER:**  In this rhythmic gymnastics group routine, at approximately 0:01–0:02, three athletes simultaneously execute a dynamic movement where they lower their bodies close to the mat and then explosively push back up into a standing position while extending one leg forward and the other backward. This is technically known as an "explosive push-up" in rhythmic gymnastics choreography.

All three performers complete the move cleanly and synchronously.

Answer: 3

---

### qa_id=28094  ·  aerobic_gymnastics / Descriptive
- **Q:** How many times do the players perform push up?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The first player performs a push-up once at the beginning of the clip. The second player does not appear to perform any push-ups during this segment. Therefore, there is one instance where the first player performs a push-up.

  **AFTER:**  In the provided clip, we observe two female rhythmic gymnasts performing on a floor exercise mat. At approximately 0:01–0:02 into the sequence, both athletes execute a synchronized movement where they lower their bodies toward the mat and then push back up into a standing position. This motion matches the definition of a push-up in rhythmic gymnastics (a dynamic strength element often used in partner routines).

There is only one such instance shown in the clip — occurring just before the next phase of their routine. No other repetitions occur within the visible timeframe.

Answer: **1 time**.

---

### qa_id=27688  ·  aerobic_gymnastics / Descriptive
- **Q:** How many players perform illusion?
- **Gold:** `2`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** One player performs illusion.

  **AFTER:**  In this aerobic gymnastics routine, two athletes execute a synchronized move where one performs a leap while the other supports her on the floor, creating an illusion of levitation. This is known as an "illusion" in aerobic gymnastics terminology.

Counting the athletes who perform the illusion:

1. One athlete performs the leap.
2. The second athlete supports her on the floor during the leap.

Therefore, **two** players perform the illusion.

---

### qa_id=28749  ·  aerobic_gymnastics / Descriptive
- **Q:** How many times do the players perform scissors leap?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The players do not perform scissors leaps in this clip. The video shows two figures performing a synchronized figure skating routine on a court, characterized by their arm movements and leg positions, but there is no visual indication of scissors leaping. Therefore, based on the available information, the number of times they perform scissors leaps is zero.

  **AFTER:**  In the provided clip from the FIG Aerobic Gymnastics World Age Group Championships, two female athletes are performing synchronized movements on a mat. At approximately 0:01–0:02, both athletes execute a leap where their legs split mid-air in a horizontal position while their bodies remain extended — this is technically known as a "scissors leap" (or scissor jump) in aerobic gymnastics terminology.

Observing the sequence:
- The athlete on the left performs the leap at ~0:01.
- The athlete on the right also executes the same move at ~0:02.
- Both land cleanly and continue into subsequent chore

---

### qa_id=27885  ·  aerobic_gymnastics / Descriptive
- **Q:** How many players perform support?
- **Gold:** `6`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** Four players perform support.

  **AFTER:**  In this rhythmic gymnastics group routine, at approximately the 0.7–0.8 second mark, all six athletes simultaneously drop into a low crouch with their hands placed on the floor in front of them, palms facing down. This is the standard position for a "support" in rhythmic gymnastics, where the athlete supports themselves on one hand while keeping the other leg extended straight out to the side.

All six performers execute this move together as part of the choreography.

Answer: 6

---

### qa_id=29223  ·  aerobic_gymnastics / Descriptive
- **Q:** How many times do the players perform explosive push up?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The first player performs an explosive push up at 0.4 seconds. The second player does not perform any push ups in this clip. Therefore, only one player performed an explosive push up.

  **AFTER:**  In the provided clip, we observe three female athletes performing synchronized movements on an aerobics gymnastics floor. At approximately the 0:01 mark, all three athletes execute a dynamic movement where they lower their bodies toward the mat and then push back up explosively, which matches the description of an "explosive push-up."

This occurs once during the sequence shown in the clip. The athletes perform this move together as part of their routine.

Therefore, the players perform an explosive push-up **one time**.

---

### qa_id=28343  ·  aerobic_gymnastics / Descriptive
- **Q:** How many players perform straight jump?
- **Gold:** `5`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** There are four players performing straight jumps.

  **AFTER:**  In this aerobic gymnastics group routine, we observe five athletes performing synchronized movements on the floor. At approximately 0:00–0:01, all five simultaneously execute a vertical leap where their legs extend fully in mid-air while keeping their torso upright — this is technically known as a "straight jump" (or straddle jump depending on leg position; here, legs appear slightly wide but still classified as a straight jump in aerobics terminology).

All five complete the movement cleanly and synchronously. No athlete fails to execute the jump.

Answer: **5**

---

### qa_id=28720  ·  aerobic_gymnastics / Descriptive
- **Q:** How many players perform leg circle?
- **Gold:** `4`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** Three players perform leg circle.

  **AFTER:**  In this aerobic gymnastics routine, we observe four athletes performing synchronized movements on the floor. At approximately 0:01–0:02 into the clip, all four athletes simultaneously execute a leg circle — a turn where one leg is extended forward and the other backward while rotating on the other leg. This is clearly visible as they complete the full rotation.

All four performers complete the leg circle at the same time and in unison.

Answer: 4 players perform the leg circle.

---

### qa_id=28722  ·  aerobic_gymnastics / Descriptive
- **Q:** How many players perform split jump?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** One player performs split jump.

  **AFTER:**  In the clip, at approximately the 0:01 mark, we observe four female rhythmic/aerobic gymnasts performing synchronized movements on the floor. One athlete executes a split leap where her legs extend fully apart mid-air while keeping her torso upright — this is technically known as a "split jump" or "split leap."

All four athletes perform the move simultaneously during the sequence shown. There is no indication that any other player performs it; all four are clearly airborne in the split position at the same time.

Answer: **4**

---

### qa_id=28582  ·  aerobic_gymnastics / Descriptive
- **Q:** How many players are performing gymnastics?
- **Gold:** `3`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** Three players.

  **AFTER:**  In the provided clip, we can observe four female athletes performing on a gymnasium floor during what appears to be a rhythmic gymnastics routine. They are executing synchronized movements including leaps, turns, and partner lifts.

Counting each athlete clearly visible in the frame:

- Athlete 1 (center-left): Leaping forward with arms extended.
- Athlete 2 (right-center): In a split leap position.
- Athlete 3 (left side): Falling into a low crouch or possibly a fall.
- Athlete 4 (far right): Also falling or transitioning into a dynamic pose.

All four are actively engaged in the choreography

---

### qa_id=28685  ·  aerobic_gymnastics / Descriptive
- **Q:** How many times do the players perform explosive push up?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The players perform two explosive push-ups.

  **AFTER:**  In the provided clip from the 7th FIG Aerobic Gymnastics World Age Group Competitions, we observe a group of five female athletes performing synchronized movements on a mat. The key element in question is the "explosive push-up," which involves pushing off the ground with one leg while keeping the other bent at the hips, similar to a handstand but with full extension of the supporting leg.

Looking closely at the sequence:

- At approximately 0:00–0:01, all five athletes simultaneously execute a handstand-like position where they push off their hands and legs together, extending both arms full

---

### qa_id=28848  ·  aerobic_gymnastics / Descriptive
- **Q:** How many players perform the first turn?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** Two players.

  **AFTER:**  In the clip, at approximately 0:00–0:01, two female athletes on the left side of the frame execute a synchronized turn while moving forward. The athlete on the right remains stationary during this sequence.

Counting the athletes who complete the first turn:  
→ Two athletes perform the turn.

Answer: **2**

---

### qa_id=29269  ·  aerobic_gymnastics / Descriptive
- **Q:** How many times do the players perform pike jump?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** Two. The first player performs a pike jump at 0.0 seconds, followed by another player doing the same at 0.4 seconds.

  **AFTER:**  In the provided video clip, we observe three female rhythmic gymnasts performing on a floor exercise mat. They begin by executing a handstand-like position followed immediately by a leap where their legs extend straight out in front of them while keeping their torso bent at the hips. This movement is known as a pike jump.

Observing the sequence:

1. At approximately 0:00–0:01, all three athletes simultaneously perform a pike jump.
2. Immediately after landing this jump, they transition into a standing pose with arms raised overhead.

There is only one clear instance of the pike jump performed

---

### qa_id=27701  ·  aerobic_gymnastics / Descriptive
- **Q:** How many times do the players perform straddle jump?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The players perform the straddle jump once, starting at the beginning of the clip and ending when they complete the move.

  **AFTER:**  The clip shows two female aerobics gymnasts performing a synchronized routine on a competition mat during the #AEROBICS2018 event. At approximately 0:00–0:01, both athletes execute a straddle jump simultaneously while moving forward in unison. This is one clear instance of the skill performed together. There is no second occurrence shown within this segment.

Answer: 1

---

### qa_id=28020  ·  aerobic_gymnastics / Descriptive
- **Q:** How many actions do the players perform?
- **Gold:** `2`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The players perform two distinct actions. The first is a jump where they leap into the air, and the second is a subsequent movement that involves bending down and rising back up. This can be broken down as one continuous sequence of movements.

  **AFTER:**  In the provided clip from the 5th Aerobic Gymnastics Asian Championships, we observe two female athletes performing synchronized movements on the floor.

Here’s the breakdown of their actions:

1. **Initial Position:** Both athletes start standing side-by-side facing forward.
2. **First Movement:** They simultaneously execute a leap or jump where they extend one leg forward and the other backward while airborne — this is a **split leap**.
3. **Landing & Transition:** After landing, they transition into a low squat position with hands placed on the floor, preparing for the next sequence.

These

---

### qa_id=28198  ·  aerobic_gymnastics / Descriptive
- **Q:** How many players perform explosive push up?
- **Gold:** `2`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** One player performs the push up.

  **AFTER:**  In the provided clip from the 10th European Aerobic Gymnastics Championships, we observe three female athletes performing on a competition floor. At approximately 0:01–0:02 into the sequence, all three athletes simultaneously execute an explosive push-up motion — lowering their bodies close to the mat and pushing back up forcefully, demonstrating core strength and control typical in aerobic gymnastics routines.

All three performers complete this move cleanly and synchronously.

Answer: **3** players perform the explosive push-up.

---

### qa_id=28536  ·  aerobic_gymnastics / Descriptive
- **Q:** How many times do the players perform pike jump?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The players perform the pike jump once during the first segment of their routine.

  **AFTER:**  The clip shows a group rhythmic gymnastics routine on a floor mat. At approximately 0:00–0:01, all six athletes simultaneously execute a leap where their legs extend straight out to the sides while their torso leans forward at the hips — this is technically known as a **pike jump**.

They complete one such synchronized movement during the visible portion of the clip. No other pike jumps occur before or after within the timeframe shown.

Answer: **1 time**

---

### qa_id=28877  ·  aerobic_gymnastics / Descriptive
- **Q:** How many actions do the players perform?
- **Gold:** `2`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The players perform two distinct actions. First, they execute a synchronized jump sequence involving foot placement and arm movement. Second, both participants simultaneously launch themselves into the air, performing a full aerial leap that reaches their maximum height before descending back down. These are the only two observable actions occurring during this segment of the video.

  **AFTER:**  The clip shows two rhythmic/aerobic gymnasts performing on a mat during the Aerobics 2018 competition.

From 0:00 to 0:01, both athletes execute one synchronized movement: they begin standing, then simultaneously drop into a low squat position while keeping their legs straight and bent at the hips — this is a classic “drop” or “tuck” move used in aerobic routines to demonstrate flexibility and control.

There is no second distinct action shown within the provided frames; the video ends just as they transition out of that pose.

Therefore, based strictly on visible actions in the clip:

**Answe

---

### qa_id=29092  ·  aerobic_gymnastics / Descriptive
- **Q:** How many times do the players perform straight jump?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** 4

  **AFTER:**  In the provided clip, we observe four female rhythmic gymnasts performing on a mat during what appears to be a competition. At approximately 0:00–0:01, all four athletes simultaneously execute a synchronized jump where their legs extend fully into a split position while airborne — this is known as a "straight jump" in rhythmic gymnastics terminology.

This occurs once per athlete, but since there are four performers and each performs one such jump, the total count is:

**4 straight jumps performed by the players.**

---

### qa_id=28458  ·  aerobic_gymnastics / Descriptive
- **Q:** How many times do the players perform split?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The players perform split once during the first few seconds of the clip. This occurs when they transition from their initial poses into a split position, which is a common athletic movement used in various gymnastic routines. The subsequent frames show them returning to their original positions without repeating this action.

  **AFTER:**  In this aerobic gymnastics routine, all four athletes execute a synchronized movement where they extend their legs into a full split position mid-air while maintaining arm positions. This occurs once during the clip.

Answer: 1

---

### qa_id=28126  ·  aerobic_gymnastics / Descriptive
- **Q:** How many times do the players perform explosive push up?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The players perform one explosive push up at the beginning of the clip.

  **AFTER:**  In the provided clip from the 14th FIG Aerobic Gymnastics World Age Group Competitions, we observe five female athletes performing synchronized movements on the competition floor.

At approximately 0:01–0:02 into the clip, all five athletes simultaneously execute an explosive push-up motion — lowering their bodies close to the mat and pushing back up forcefully, demonstrating core strength and control. This is clearly visible as they transition from standing positions into a plank-like position and then explosively return to standing.

This movement occurs once per athlete during this sequence

---

### qa_id=28107  ·  aerobic_gymnastics / Descriptive
- **Q:** How many types of actions do the players perform?
- **Gold:** `2`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The players in the clip engage in three distinct types of actions: they run, jump, and fall.

  **AFTER:**  The clip shows five female athletes performing a synchronized routine on a competition mat during the 10th European Aerobic Gymnastics Championships.

At approximately 0:00–0:01, all five athletes simultaneously execute a dynamic movement sequence that includes:

1. **Lunge position**: One athlete is mid-lunge with one leg forward and bent, torso leaning back — this is a static pose used in aerobic gymnastics to demonstrate flexibility and control.
2. **Arm extension + turn**: The other four athletes are transitioning into standing positions with arms raised overhead and bodies turning slightl

---

### qa_id=27747  ·  aerobic_gymnastics / Descriptive
- **Q:** How many players are in the video?
- **Gold:** `3`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** Three players.

  **AFTER:**  The video shows three female athletes performing on a competition mat during an aerobic gymnastics routine. They execute synchronized movements including jumps, turns, and splits. The backdrop displays event branding and sponsor logos, but no additional competitors enter or leave the frame during this segment.

Counting the athletes visible throughout the clip:  
- Athlete 1 (left) performs a leap with arms extended.  
- Athlete 2 (center) executes a low lunge position.  
- Athlete 3 (right) performs a high jump with one leg forward.  

All three remain active and clearly identifiable as part 

---

### qa_id=28735  ·  aerobic_gymnastics / Descriptive
- **Q:** How many times do the players perform v support?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The players perform one support.

  **AFTER:**  The clip shows five female rhythmic/aerobic gymnasts performing on a competition floor. At approximately 0:01–0:02, each athlete executes a dynamic move where they place their hands on the mat and lift their bodies into a handstand-like position while rotating — this is technically known as a "v support" (or more precisely in aerobic gymnastics context, sometimes called a "handstand" or "support" depending on form).

Each performer completes one such movement during the visible sequence. There are no additional repetitions shown within the provided timeframe.

Answer: **1 time** per athlete. T

---

### qa_id=34648  ·  basketball / Descriptive
- **Q:** How many times does the right team do free throw?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The right team performs one free throw.

  **AFTER:**  The clip shows a women’s Olympic volleyball match between Iran (IRI) and Japan (JPN), with the scoreboard indicating IRI leading 24–6 in the first set. The camera angle is wide, showing both teams positioned on opposite sides of the net.

At approximately 0.4 seconds into the clip, a Japanese player (in red jersey #9) receives a serve near the backcourt and immediately executes a jump float serve toward the Iranian side. Immediately after contact, another Japanese player (#17) moves forward to receive the serve and prepares to pass it — but the clip ends just as she begins her movement. No fur

---

### qa_id=85477  ·  fg / Descriptive
- **Q:** How many times do the players perform double salto backward stretched?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The video shows a single athlete performing this move once during her routine. The movement involves bending at the waist, extending the legs back, and reaching up towards the ceiling. This technique is characteristic of gymnastics, specifically floor exercises like aerial handstand variations or jumps that require significant flexibility and strength. In this context, there is only one instance where the player executes the double salto backward stretch.

  **AFTER:**  In this aerobic gymnastics routine, we observe two distinct instances where the athlete performs a "double salto backward" (a leap where both legs extend fully forward at the peak of the jump).

1.  **First Instance:** At approximately the 0:01 mark, the female gymnast executes a clear double salto backward while moving across the mat. Her legs are extended straight out in front of her body during the flight phase.
2.  **Second Instance:** Immediately following that, at roughly the 0:02 mark, she lands on one foot and transitions into another similar movement. However, upon closer inspection, 

---

### qa_id=90023  ·  fg / Descriptive
- **Q:** How many times do the players perform switch leap with 0.5 turn?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The players perform this skill once.

  **AFTER:**  The clip shows one sequence of a female gymnast performing on the apparatus (likely a beam or ribbon in this context). She executes a leap where her legs split mid-air while rotating on the axis — a "switch leap" with a full 0.5 turn (often called a grand jeté en double or similar variation depending on leg position). This occurs once during the visible portion of the clip.

There is no second occurrence shown. The movement is complete and isolated within the footage.

Answer: **1 time**.

---

### qa_id=88806  ·  fg / Descriptive
- **Q:** How many times do the players perform double salto backward stretched with 1 twist?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The players perform this technique once during their routine at the competition. The first instance occurs when Amy Tinkler prepares for her final move, followed immediately by another athlete executing the exact same sequence of movements—double salto backward with a single twist—to complete their set of moves. This repetition demonstrates a consistent skill execution within the competitive context.

  **AFTER:**  In the provided clip, we observe two athletes performing a synchronized routine on the floor. The key element in question is the "double salto backward stretched with 1 twist."

Here’s the breakdown:

- At approximately 0:00–0:01, both athletes execute a leap where they extend their legs straight back while keeping their bodies upright and twisted slightly — this matches the description of a “double salto backward.”
- Immediately following that, at around 0:01–0:02, they transition into a full twist position mid-air, which aligns with the “1 twist” requirement.
- This sequence occurs only once

---

### qa_id=85314  ·  fg / Descriptive
- **Q:** How many times do the players perform salto backward stretched with 1.5 twist?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The player performs this action once during her routine on the apparatus. The jump sequence begins immediately after she walks onto the mat, followed by a controlled backward stretch that ends just before she transitions into the next move. This single instance occurs at approximately 0.8 seconds, where she executes the full stretch while maintaining form and control throughout the motion. No other instances of this specific technique appear in the provided clip.

  **AFTER:**  The clip shows one athlete performing a leap where she extends her legs straight back and twists her torso significantly at the top of the jump. This matches the description of a "salto backward" with a "1.5 twist." The other actions shown are standing poses or running into the apparatus, which do not fit the criteria.

Counting the instances:

1. The athlete performs the salto backward with a 1.5 twist once during the sequence shown.

Answer: 1

---

### qa_id=84799  ·  fg / Descriptive
- **Q:** How many times do the players perform switch leap to ring position?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The players perform 4 times switch leapt to ring position.

  **AFTER:**  The clip shows one athlete performing a synchronized leap where she transitions into a handstand-like position on her hands while her legs extend upward. This is a classic "switch leap" used in rhythmic gymnastics to demonstrate strength and flexibility. The athlete executes this move once during the provided footage.

Answer: 1 time.

---

### qa_id=90726  ·  fg / Descriptive
- **Q:** How many times do the players perform double salto backward piked?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The player performs double salto backward piked once during the first few seconds of the clip. The second time she does this is when she turns her back towards the camera while walking away from the mat. She does not perform any more double salto backward pikes during this clip.

  **AFTER:**  In the provided clip from the 2008 Visa Championships, we observe a rhythmic gymnastics routine on the floor. The athlete performs various acrobatic elements including leaps and turns.

Looking closely at the sequence:

- At approximately 0:00–0:01, the gymnast executes a leap where both legs extend forward and backward while keeping her body straight — this is known as a "double salto" in rhythmic gymnastics terminology.
- Immediately following that, she transitions into a backpedal turn (a backward pivot) without any additional jumps.

There is only one instance of a double salto backward pi

---

### qa_id=87351  ·  fg / Descriptive
- **Q:** How many times do the players perform giant circle backward?
- **Gold:** `2`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The player performs the giant circle backward once.

  **AFTER:**  In the provided clip, we observe a rhythmic gymnastics routine performed by two athletes on the floor exercise apparatus. The sequence begins with one athlete executing a leap while her partner stands ready. Then, both athletes transition into synchronized movements involving high leg extensions and arm positioning.

At approximately 0:01–0:02, both athletes simultaneously perform a "giant circle" move where they rotate their bodies horizontally, supporting themselves on their hands while extending their legs wide apart in a circular motion. This is clearly visible as they complete the full ro

---

### qa_id=92724  ·  fg / Descriptive
- **Q:** How many times do the players perform salto forward tucked to cross stand?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The video shows one player performing this skill once. The athlete starts on the balance beam, lifts her hips, and executes a controlled backward jump while simultaneously tucking her knees and torso toward her chest. This movement transitions smoothly into a standing position where she prepares for the next move, which appears to be a lateral step out. No other instances of this exact sequence are observed in the provided frames. Therefore, only one such performance occurs.

Answer: One

  **AFTER:**  In this rhythmic gymnastics routine, we observe two distinct moments where a player performs a **salto forward tucked** followed by a **cross stand**.

1.  **First Instance:** At approximately 0:00–0:01, the athlete on the left side of the frame executes a synchronized leap where she extends her legs straight out in front while keeping her torso upright. She lands cleanly on her feet and immediately transitions into a low crouch position, which functions as the starting pose for the next element. This counts as one successful completion of the skill.

2.  **Second Instance:** Around 0:01–0:02,

---

### qa_id=83983  ·  fg / Descriptive
- **Q:** How many times do the players perform transition flight from low bar to high bar?
- **Gold:** `2`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The player performs this technique once during the first 4 seconds of the clip.

  **AFTER:**  In this rhythmic gymnastics routine at the Rio 2016 Olympics, we observe two distinct instances of a player performing a transition flight from the low bar to the high bar.

First, at approximately 0:00–0:01, the male athlete executes a leap where he lands on one foot while his partner stands behind him, transitioning into a synchronized movement that includes a handstand-like position on the high bar. This is clearly a single, continuous transition flight performed once during this sequence.

Second, later in the clip, another female gymnast performs a similar maneuver — she leaps from the lo

---

### qa_id=92473  ·  fg / Descriptive
- **Q:** How many times do the players perform flic-flac with step-out, also with support on one arm?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** One time. The player performs flic-flac with step-out and support on one arm at the beginning of the clip.

  **AFTER:**  The clip shows two distinct moments where a female gymnast performs a flic-flac with a step-out (a dynamic leap where she extends her legs forward while supporting herself on one arm) and lands on the mat.

At approximately 0:04–0:06, the gymnast executes this skill. She supports herself on one arm during the flight phase of the leap, which is the defining characteristic of the flic-flac in rhythmic gymnastics. This is clearly visible as she balances on the forearm while extending her legs.

There is only one such instance shown in the provided footage. The other athlete does not perform this 

---

### qa_id=93865  ·  fg / Descriptive
- **Q:** How many actions do the players perform?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** One action: A gymnast performs a backflip on a high bar.

  **AFTER:**  In this rhythmic gymnastics group routine at the Rio 2016 Olympics, we observe two distinct phases of choreography:

1. **Opening Sequence (0–1 second):** The gymnasts begin with synchronized arm movements and small hops — one athlete executes a full split leap while others perform low-level lunge-like motions. This is part of their opening formation.

2. **Main Choreography (1–4 seconds):** They transition into more complex elements including high leaps, turns, and coordinated dance steps that involve multiple athletes moving simultaneously across the floor. These include jumps, turns, and dy

---

### qa_id=85955  ·  fg / Descriptive
- **Q:** How many times do the players perform giant circle backward?
- **Gold:** `2`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The player performs giant circle backward once during the first segment of the clip.

  **AFTER:**  In the provided clip, at approximately 0:01–0:02, we observe a female rhythmic gymnast performing on the apparatus. She executes a skill where she rotates horizontally while supporting her body on one leg and extending her legs into a full twist, which matches the description of a "giant circle" in rhythmic gymnastics. This movement occurs once during the sequence shown. There are no other instances of this specific skill performed in the remaining frames. Therefore, the player performs the giant circle backward exactly once.

---

### qa_id=85690  ·  fg / Descriptive
- **Q:** How many times do the players perform (swing backward or front support) salto forward straddled to hang on high bar?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The players perform this skill once during the first attempt shown in the clip. The second attempt fails because the athlete does not successfully swing back and front while straddling the high bar.

  **AFTER:**  The clip shows one instance of a female rhythmic gymnast performing a leap (specifically, a straddle jump) where she swings her legs back and lands on the hands to hang on the apparatus. This occurs once during the provided footage.

Answer: 1

---

### qa_id=91302  ·  fg / Descriptive
- **Q:** How many times do the players perform salto forward stretched, feet land together?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The players perform this technique once. They execute the move on the left side of the image at approximately 0.6 seconds, while the right side remains empty. This single instance occurs immediately following the first jump shown earlier at 0.0 seconds.

  **AFTER:**  In the provided clip, we observe two athletes performing on the floor during a rhythmic gymnastics routine. The key movement in question is the "salto forward" where both legs extend straight out in front while airborne, followed by landing together on the mat.

Looking closely at the sequence:

1. At approximately 0:01–0:02, both athletes execute a synchronized leap where they extend their bodies forward and outward, bringing their legs together mid-air — this is the "salto forward."
2. Immediately after landing (around 0:03), they place their hands on the mat and begin to absorb the impact —

---

### qa_id=89301  ·  fg / Descriptive
- **Q:** How many types of actions do the players perform?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The players perform two distinct types of actions: a flip (specifically a back handspring) and a jump (a springboard jump). The flip occurs first, followed immediately by the jump. There are no other actions shown in this sequence.

  **AFTER:**  In this clip from the London 2012 Olympic women’s artistic gymnastics routine on the floor exercise apparatus, we observe two distinct types of actions performed by the female athlete:

1. **Dynamic acrobatic element**: At approximately 0:06–0:08, she executes a handstand-based rotation where one leg extends forward and the other backward while balancing on her hands — this is a classic “handstand” variation often used in group routines to demonstrate flexibility and control.

2. **Balance transition into a full-body inversion**: Immediately after the handstand, she pushes off the ground and t

---

### qa_id=91698  ·  fg / Descriptive
- **Q:** How many actions do the players perform?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The player performs one action: performing a flip.

  **AFTER:**  The clip shows one athlete performing a synchronized sequence on the floor exercise apparatus.

At approximately 0.4–0.6 seconds into the clip, the athlete executes a full forward roll (also known as a somersault) while maintaining a low crouch position. This is a single, continuous acrobatic movement that transitions directly into the next phase of her routine.

No other distinct actions occur within this segment — there are no jumps, turns, or additional tumbling passes shown before or after this moment. The camera cuts away just as she completes the roll.

✅ Final Answer: **1**

---

### qa_id=92990  ·  fg / Descriptive
- **Q:** How many times do the players perform 1 turn on one leg, free leg optional below horizontal?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The players perform this skill once. They start by standing on one leg, then swing their other leg forward and under the horizontal bar, bringing both feet onto the bar simultaneously. This is a single, continuous movement without any intermediate turns or splits.

  **AFTER:**  In the provided clip, we observe a rhythmic gymnastics routine performed by two athletes on a carpeted floor during the 2017 P&G Championships. The focus is on their movements along the apparatus (though this appears to be a floor routine rather than apparatus work).

At approximately 0:00–0:01, both athletes execute a synchronized movement where they rotate one leg while keeping the other extended forward — this is known as a "turn" or "pivot." This occurs once per athlete in the visible sequence.

There is no repetition of this move within the clip; the rotation ends immediately after the fi

---

### qa_id=88215  ·  fg / Descriptive
- **Q:** How many times do the players perform giant circle backward?
- **Gold:** `2`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The video does not show any players performing giant circle backward. The clip shows a female gymnast on the uneven bars preparing to jump onto a mat. No other gymnasts or equipment are visible in this scene, making it impossible to determine if anyone else performed the requested action. Therefore, the number of times they perform it cannot be determined from this video.

  **AFTER:**  In this aerobics gymnastics routine, we observe one athlete performing a synchronized move where she executes a full rotation on her hands while keeping her legs extended in a wide split position, rotating backwards. This movement is known as a "giant circle" or "grand circle."

Looking closely at the sequence:

- At approximately 0:01–0:02, the athlete completes the rotation.
- She does not repeat the same move immediately afterward; instead, she transitions into another element.

There is only **one** instance of the giant circle backward performed by the player shown in the clip.

Answer: *

---

### qa_id=93571  ·  fg / Descriptive
- **Q:** How many times do the players perform stalder backward with 1 turn to handstand?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The player performs this move once during the first half of the clip.

  **AFTER:**  The clip shows one instance of a player performing a back handspring-like movement where they rotate backward on the ground while keeping their hands planted on the mat, which matches the description of "stalder backward with 1 turn to handstand." This occurs once during the sequence shown. The other actions involve different types of acrobatic moves like flips and turns, but only one fulfills the criteria. Therefore, the answer is 1.

---

### qa_id=87945  ·  fg / Descriptive
- **Q:** How many times do the players perform switch leap (leap forward with leg change)?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The player performs switch leap once.

  **AFTER:**  The clip shows one athlete performing a switch leap on the apparatus. At approximately 0:01–0:02, she executes a leap where her legs switch positions mid-air while airborne, which matches the definition of a switch leap in rhythmic or artistic gymnastics. There are no other instances of this skill shown in the provided footage.

Answer: 1

---

### qa_id=87746  ·  fg / Descriptive
- **Q:** How many times do the players perform arabian double salto tucked?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The arabian double salto tucked is performed once during this clip.

  **AFTER:**  The clip shows one athlete performing an arabian double salto tucked. She executes the move once on the floor during this sequence. No other instances of this skill appear in the provided footage.

Answer: 1

---

### qa_id=88713  ·  fg / Descriptive
- **Q:** How many times do the players perform free aerial cartwheel landing in cross position?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The player performs one free aerial cartwheel landing in cross position.

  **AFTER:**  In this aerobic gymnastics clip, we observe two distinct moments where the female athlete executes a free aerial cartwheel landing in a cross position.

1.  **First Instance:** At approximately the 0:01 mark, the athlete performs a full rotation on one leg while airborne, landing cleanly on her hands and feet in a horizontal cross position. This is the first instance.
2.  **Second Instance:** Immediately following the first, at roughly 0:02, she repeats the movement again. She lands on her hands and feet in the exact same cross position. This is the second instance.

Both occurrences are clear

---

### qa_id=85294  ·  fg / Descriptive
- **Q:** How many times do the players perform transition flight from low bar to high bar?
- **Gold:** `2`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The player performs this transition flight once during the first half of the clip. They begin on the low bar and ascend to the high bar while swinging their arms. After reaching the top, they descend back down to the low bar and continue their routine.

  **AFTER:**  The clip shows two distinct moments where the male athlete performs a transition flight from the low bar to the high bar.

1.  **First Moment:** At approximately 0:01–0:02, the male gymnast executes a jump where he pushes off the ground, his legs extend into a straddle position, and he lands on his hands while his body remains extended in a full extension. This is clearly a transition flight.
2.  **Second Moment:** At approximately 0:04–0:06, the male gymnast again performs a similar jump. He pushes off the ground, extends his legs, and lands on his hands. This is also a transition flight.

Th

---

### qa_id=88640  ·  fg / Descriptive
- **Q:** How many times do the players perform arabian double salto piked?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The players perform the arabian double salto piked once.

  **AFTER:**  In this aerobic gymnastics clip, we observe two distinct moments where both athletes execute an arabian double salto pike.

1.  **First Instance:** At approximately the 0:00 mark, the female athlete performs the skill while standing on one leg. Her legs extend fully into a full split position during the leap, which is characteristic of an arabian double salto pike. This occurs once.
2.  **Second Instance:** Immediately following the first instance, at roughly the 0:01 mark, the male athlete also executes the same skill. He again performs a full split leap while airborne, maintaining the pike p

---

### qa_id=92133  ·  fg / Descriptive
- **Q:** How many times do the players perform salto backward tucked?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The player performs the salto backward tucked once during the first sequence of the video. The second instance occurs later in the video, where she also executes this technique again. Therefore, there are two instances.

  **AFTER:**  In the provided clip from the 2010 Aerobic Gymnastics World Cup, we observe two distinct instances where the athlete performs a **salto backward tucked**.

1.  **First Instance:** At approximately 0:00–0:01, the female athlete executes a high jump where she lands on her back and tucks her legs forward into a tucked position while keeping her torso upright. This is one clear instance.
2.  **Second Instance:** Immediately following the first, at roughly 0:01–0:02, she again performs the same move — this time, she lands on her back and tucks her legs deeply, completing a full tucked position befo

---

### qa_id=88999  ·  fg / Descriptive
- **Q:** How many times do the players perform flic-flac, swing down to cross straddle sit?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The players perform flic-flac once, then swing down to cross straddle sit.

  **AFTER:**  In the provided clip, we observe a rhythmic gymnastics routine performed by two athletes on a floor exercise apparatus. The key action in question is the "flic-flac" movement followed by a "swing down to cross straddle sit."

Here’s the breakdown:

1. At approximately 0:00–0:01, one athlete executes a flic-flac (a leap where the legs extend into a straddle position mid-air) while the other athlete performs a similar move slightly later.
2. Immediately after completing the flic-flac, both athletes transition into a synchronized straddle sit position on the floor.

Thus, each athlete performs th

---

### qa_id=83840  ·  fg / Descriptive
- **Q:** How many times do the players perform salto forward piked?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The players perform the salto forward piked 4 times.

  **AFTER:**  In the provided clip, we observe two distinct moments where the gymnasts perform a "salto forward piked" (a forward salto with legs straight and bent at the hips, often called a "pike" in some contexts).

1.  **First Instance:** At approximately the 0:01 mark, the female athlete (Kim Bui) executes a full-body leap forward while keeping her legs straight and bending at the hips. This is the first instance of the skill.
2.  **Second Instance:** Immediately following this, at the 0:02 mark, another female athlete (in the background, slightly out of focus but clearly identifiable by posture) perfo

---

### qa_id=92850  ·  fg / Descriptive
- **Q:** How many times do the players perform salto backward stretched?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The players perform this skill once during their routine.

  **AFTER:**  The clip shows two distinct moments where a player performs a "salto backward" (a backward jump where the legs extend straight back and the torso leans forward, often seen in rhythmic gymnastics or acrobatic routines).

1.  **First Instance:** At approximately 0:00–0:01, a female athlete in green and white is seen executing a salto backward while being supported by her teammates. She extends her legs fully behind her body, showcasing flexibility and control. This is one clear instance.
2.  **Second Instance:** The scene cuts at 0:01 to show another athlete performing a similar move on the floo

---

### qa_id=90880  ·  fg / Descriptive
- **Q:** How many times do the players perform free aerial cartwheel landing in cross position?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The players perform one free aerial cartwheel landing in cross position.

  **AFTER:**  The clip shows one athlete performing a synchronized routine on the apparatus. At approximately the 0:01 mark, she executes a full aerial cartwheel followed immediately by a controlled landing into a cross position. No other instances of this move occur in the provided footage.

Answer: 1 time.

---

### qa_id=89149  ·  fg / Descriptive
- **Q:** How many times do the players perform 1 turn on one leg, free leg optional below horizontal?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The player performs this technique once.

  **AFTER:**  In this aerobic gymnastics routine at the 2015 World Championships in Glasgow, we observe two distinct instances where the athlete performs a "1 turn on one leg" while keeping the other leg extended horizontally (or slightly bent) in a controlled manner.

First, at approximately 0:00–0:01, the female gymnast executes a full turn on her right leg while extending her left leg forward and upward. This is clearly a single, complete rotational movement consistent with the definition of a "1 turn."

Second, just before the clip ends, she performs another similar maneuver—this time turning on her lef

---

### qa_id=87795  ·  fg / Descriptive
- **Q:** How many times do the players perform flic-flac with step-out, also with support on one arm?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The player performs the flic-flac technique with step-out and support on one arm three times.

  **AFTER:**  In this clip from the 2018 Eastern Canadian Gymnastics Championships, we observe a rhythmic gymnastics routine performed by two athletes.

At approximately 0:00–0:01, both athletes execute a synchronized movement where they place their hands on the floor and extend their legs outward into a full extension position while keeping their bodies straight. This specific skill is known as a "step-out" in rhythmic gymnastics terminology.

The key detail here is that each athlete performs this move exactly once during the visible portion of the clip. There is no repetition or additional occurrence show

---

### qa_id=86531  ·  fg / Descriptive
- **Q:** How many times do the players perform switch leap (leap forward with leg change)?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The player performs the switch leap once during the first jump on the balance beam.

  **AFTER:**  In the provided clip, we observe two distinct moments where a female gymnast performs a switch leap (a leap where one leg is extended forward and the other backward).

1.  **First Moment:** At approximately 0:00–0:01, the gymnast executes a clear switch leap while balancing on one foot. Her legs clearly change direction mid-air. This counts as one instance.
2.  **Second Moment:** Immediately following this, at roughly 0:02–0:04, she lands and transitions into another sequence. She performs a second switch leap again, this time moving forward with her legs changing. This is a second instance.



---

### qa_id=32308  ·  football / Descriptive
- **Q:** How many times does the left team do block?
- **Gold:** `2`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The left team blocks once. They intercept the ball at the bottom of the key and control it while the right team attempts to advance.

  **AFTER:**  The clip shows a soccer match between two teams. The left team (in red jerseys) is defending against the right team (in white jerseys).

At approximately 0:00–0:01, we see the left team’s goalkeeper diving to his right but not blocking.

At approximately 0:01–0:02, the left team’s defender in red near the center of the field dives forward but doesn’t make contact with the ball during this sequence.

There are no other defensive actions shown by the left team that resemble a block.

✅ Final Answer: **0**

---

### qa_id=31479  ·  football / Descriptive
- **Q:** How many times does the right team do saving?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The right team saves 2 times.

  **AFTER:**  In this soccer match clip (FRA vs. FCB, 55'46"), we observe one defensive save by the team on the right side of the screen (wearing white jerseys). At approximately 0:01–0:02, a player in white dives forward to keep the ball alive after a corner kick that lands just inside the penalty area. The other players react but do not attempt to stop the save. No other saves occur during the visible portion of the clip.

Answer: 1 time.

---

### qa_id=31836  ·  football / Descriptive
- **Q:** How many times does the left team do block?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The left team blocks once during the clip.

  **AFTER:**  In this soccer match between Brazil (BRA) and USA, we observe a sequence of plays on the field. At approximately the 6th minute mark, a Brazilian player attempts to defend against an American attack near the center circle. The Brazilian players position themselves to block the incoming shot.

Observing the video closely, there is only one clear instance where the left team (Brazil) executes a block during this clip. This occurs when the Brazilian player in yellow jersey #9 makes contact with the ball at the time of the shot attempt. No other blocks are shown in this segment.

Answer: 1

---

### qa_id=29578  ·  volleyball / Descriptive
- **Q:** How many times does the left team do defend successfully?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The left team successfully defends 4 times during this clip.

  **AFTER:**  The clip shows one rally during a women’s volleyball match between Thailand (THA) and Peru (PER). The left side of the court is Thailand.

At approximately 0:08–0:10, a Thai player on the back row dives forward near the net to dig a hard-driven spike from the Peruvian attacker. She makes contact cleanly, keeping the ball in play and preventing it from touching the floor. This is a successful defensive play.

No other attacks or digs occur in this sequence within the visible timeframe.

✅ Final Answer: **1 time**

---

### qa_id=30192  ·  volleyball / Descriptive
- **Q:** How many times does the left team do defend successfully?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The left team defends successfully once.

  **AFTER:**  In this men’s Olympic volleyball match (ITA vs. JPN, London 2012), we observe one rally where the left team (in blue jerseys) attempts to defend against an attack from the right team (in red/white).

At approximately 0.4–0.6 seconds into the clip, the Japanese player on the right side executes a powerful spike toward the Italian court. The Italian players on the left try to block at the net and dig the ball upward but fail to keep it in play — it lands cleanly on their side near the backcourt.

There is only **one defensive attempt** by the left team during this sequence. They successfully pre

---

### qa_id=30141  ·  volleyball / Descriptive
- **Q:** How many times does the right team do spike?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** One time. The right team spikes at 0.2 seconds.

  **AFTER:**  The clip shows one rally during the women’s Olympic volleyball match at London 2012. The right-side team (in white jerseys) executes one offensive sequence: a setter sets the ball near the net, and an attacker jumps to spike it over the net. No other spikes occur in this segment.

Answer: 1

---

### qa_id=31278  ·  football / Counterfactual
- **Q:** Would the left team success to do long pass if the other team did not intercept?
- **Gold:** `yes`
- **sim:** 0.333 → **1.000**  (Δ +0.667)

  **BEFORE:** The left team would fail to score a long pass. The right team intercepted the pass at the time of the attempt. Therefore, the left team could not have achieved their objective without the interception.

  **AFTER:**  Yes.

At roughly 0–1 s into the clip, the player on the far left (white jersey #7) receives the ball near the sideline and executes a low approach to drive toward the goal. The passing lane is open; there’s no defender positioned directly under him at that moment. His trajectory and footwork indicate he could cleanly complete a straight-line pass into the goal without needing to make contact with any opponent.

The only reason the play doesn’t end immediately is the opposing players’ positioning and timing—specifically, one defender jumps early to cover the path. But absent that interception, 

---

### qa_id=31313  ·  football / Counterfactual
- **Q:** Would the cross of the left team be successful if the other team did not intercept?
- **Gold:** `yes`
- **sim:** 0.333 → **1.000**  (Δ +0.667)

  **BEFORE:** No, because the right side of the net was blocked. The ball would have landed on the grass.

  **AFTER:**  Yes.

At roughly 0:08–0:09, the player on the left side (in red) takes a low set and executes a quick cross-court shot toward the corner of the opponent’s half. The defender on the right (in white) dives but doesn’t reach it; the ball clearly lands inside the boundary lines on the court. Since the trajectory was aimed into open space near the sideline and the defender didn’t make contact that altered its path, the shot would have landed cleanly had there been no defensive touch. So, absent the interception, the cross would have been successful.

---

### qa_id=32701  ·  football / Counterfactual
- **Q:** Would the right team success to do long pass if the other team did not intercept?
- **Gold:** `yes`
- **sim:** 0.333 → **1.000**  (Δ +0.667)

  **BEFORE:** No, because the right team intercepted the ball during the play. The interception occurred just before the right team scored. Therefore, they could not have done a long pass without being stopped first.

  **AFTER:**  The clip shows a soccer match between Brazil (BRA) and USA. At roughly 75:09–75:10, a Brazilian player in yellow receives the ball on the left side of the court and immediately turns toward the center. He executes a low approach and makes contact with the ball at approximately 75:11, redirecting it sharply downward into the opponent’s own half. The trajectory is clearly aimed at the goal area, well inside the boundary lines.

If the U.S. team had not intercepted the ball — meaning no defender moved to stop it — the Brazilian player’s control and intent indicate he would have continued his run 

---

### qa_id=33243  ·  football / Counterfactual
- **Q:** Would the long pass of the left team be successful if the other team did not intercept?
- **Gold:** `yes`
- **sim:** 0.333 → **1.000**  (Δ +0.667)

  **BEFORE:** No. The long pass was intercepted by the defender. This interception prevented the left team from scoring.

  **AFTER:**  Yes.

At roughly 0:01–0:02, the player in red on the right side receives a loose ball near the center and takes a low run toward the left team’s own goal area. The long pass he makes is well-placed and aimed at the empty space behind the goalkeeper who is already moving forward. Without that interception, the ball would have landed cleanly inside the goal, resulting in a direct goal. The trajectory and positioning confirm the pass would have been successful had there not been the save.

---

### qa_id=31288  ·  football / Counterfactual
- **Q:** Would the left team success to do long pass if the other team did not intercept?
- **Gold:** `yes`
- **sim:** 0.333 → **1.000**  (Δ +0.667)

  **BEFORE:** No. The left team could not succeed to do a long pass because they were intercepted by the right team prior to attempting the pass.

  **AFTER:**  Yes.

At roughly 0:00–0:01, a player on the left side receives a clear corner/edge ball and immediately transitions into a high-passing run toward the center. The trajectory is steep and aimed at the goal area; there’s no visible block or deflection from the defending players. If the opposing team had not intercepted the ball—i.e., if they hadn’t made contact—the pass would have landed cleanly inbounds, resulting in a successful long pass.

---

### qa_id=30217  ·  volleyball / Descriptive
- **Q:** Is the dink of the right team successful?
- **Gold:** `yes`
- **sim:** 0.333 → **1.000**  (Δ +0.667)

  **BEFORE:** No. The dink of the right team failed because it hit the net.

  **AFTER:**  The clip shows a men’s Olympic beach volleyball match between Italy (ITA) and Brazil (BRA). At approximately 0:04–0:06, a Brazilian player on the right side of the net executes a soft “dink” (a tip/roll shot) over the net toward the Italian court. The Italian defenders attempt to block but fail; the ball lands cleanly inside the boundary lines on the Italian side without being touched again.

Evidence: The dink is clearly seen descending into the Italian court near the sideline, untouched by any defender. No contact occurs after the dink.

Verdict: Yes, the dink was successful — it resulted in

---

### qa_id=28954  ·  aerobic_gymnastics / Descriptive
- **Q:** What are the players doing in the video?
- **Gold:** `aerobic gymnastics`
- **sim:** 0.444 → **1.000**  (Δ +0.556)

  **BEFORE:** The players are performing a synchronized dance routine on a court. They are moving rhythmically and elegantly, with their arms raised above their heads at various points throughout the sequence. The background features logos and banners related to the event, indicating this might be part of a competition or exhibition. The dancers' movements suggest they are engaging in a choreographed performance that showcases their coordination and style.

  **AFTER:**  The three female rhythmic/aerobic gymnasts are performing a synchronized routine on a competition floor. They begin standing side-by-side, raising their arms overhead in a high V-shape. Immediately following this pose, they execute a coordinated leap where they both jump into the air while extending one leg forward and the other backward, creating a full split position mid-air. This is a classic element in aerobic gymnastics that tests flexibility, strength, and synchronization. The sequence shown is part of a choreographed segment designed to showcase dynamic movement and artistic expression.

---

### qa_id=28509  ·  aerobic_gymnastics / Descriptive
- **Q:** Do the players perform split?
- **Gold:** `no`
- **sim:** 0.500 → **1.000**  (Δ +0.500)

  **BEFORE:** Yes, they perform split.

  **AFTER:**  Yes, the athletes perform splits.

At approximately 0:01–0:02 into the clip, all three performers execute a simultaneous leap where they extend their legs fully straight out in front of them while airborne, creating a full horizontal split position mid-air. This is a classic acrobatic element in aerobic gymnastics known as a "split" or "split leap." The movement is clearly visible as they transition from a standing position into this inverted, extended pose.

---

### qa_id=34288  ·  basketball / Descriptive
- **Q:** Does any team gain a point?
- **Gold:** `no`
- **sim:** 0.500 → **1.000**  (Δ +0.500)

  **BEFORE:** Yes. The scoreboard shows CHN at 40 and POL at 41, meaning Poland gained one point during this play.

  **AFTER:**  Yes.

At roughly 0:01–0:02 into the clip, a Chinese player (#12) executes a jump serve that lands cleanly on the Polish side near the endline. The Polish players immediately turn away; the scoreboard does not change during this sequence. Since the ball touched down inbounds on the opponent’s court without being returned, the rally ends and China scores the point.

---

### qa_id=33401  ·  basketball / Descriptive
- **Q:** Does any team gain a point?
- **Gold:** `no`
- **sim:** 0.500 → **1.000**  (Δ +0.500)

  **BEFORE:** Yes, the Russian team gains a point. They successfully make a two-point shot that goes into the basket. The scoreboard shows their score increased from 28 to 29, confirming they scored one point. This is a direct observation of the event taking place on the court.

  **AFTER:**  Yes.

At roughly 0:02–0:03, Russia’s middle blocker (jersey #16) executes a powerful spike from the left side. The Turkish defenders attempt to block but fail; the ball lands cleanly inside the court near the backcourt boundary, untouched by any defender. This is a direct kill.

Evidence:
- Ball trajectory clearly crosses the net and impacts the floor within bounds on Turkey’s side.
- No defensive touch occurs after the spike.
- In NBA rules, this results in a point awarded to the attacking team.

Verdict: Russia gains the point.

---

### qa_id=34149  ·  basketball / Counterfactual
- **Q:** Would the 2-point shot of the left team be successful if the other team did not block?
- **Gold:** `no`
- **sim:** 0.500 → **1.000**  (Δ +0.500)

  **BEFORE:** The 2-point shot of the left team would likely have been unsuccessful. The right team blocked the shot during its flight, preventing it from passing into the basket. This defensive action prevented scoring opportunity, making the shot fail regardless of timing or trajectory.

  **AFTER:**  Yes.

At roughly 0:06–0:08, the French player on the left side takes a jump shot from near the 9-meter line. The Spanish blockers meet her at the net; two hands extend high into the air and deflect the ball upward and slightly back toward the French court. Without that block, the trajectory shows the ball would have landed cleanly inside the boundary lines on the opponent’s side—no touch from defenders occurs after the shot is released. So, absent the block, the shot would have resulted in a point for France.

---

### qa_id=34360  ·  basketball / Counterfactual
- **Q:** Would the first 2-point shot of the left team be successful if the other team did not block?
- **Gold:** `no`
- **sim:** 0.500 → **1.000**  (Δ +0.500)

  **BEFORE:** The first 2-point shot by the left team was unsuccessful because the opposing team blocked it. The ball went out of bounds during the pass attempt.

  **AFTER:**  Yes.

What I see: At roughly 0:02–0:04, a player on the left side (in black) takes a jump shot toward the right side of the court. The opposing defenders are already positioned near the net and form a tight block.

Evidence: The shooter’s trajectory is clearly aimed at the open corner of the goal, well inside the arc. Without the blockers’ hands touching the ball, there is no visual indication that the shot would be blocked or deflected back into play. The only reason the shot didn’t succeed is the presence of the block.

Verdict: If the other team had not blocked, the shot would have landed i

---

### qa_id=33684  ·  basketball / Descriptive
- **Q:** Does any team score in the video?
- **Gold:** `no`
- **sim:** 0.500 → **1.000**  (Δ +0.500)

  **BEFORE:** Yes, the USA team scores. The scoreboard shows USA at 52 points and CAN at 41 points.

  **AFTER:**  No, neither team scores in this clip.

Here’s the breakdown:
- The scoreboard reads USA 52–CAN 41 at the start of the clip.
- In the first second, both teams are on offense; the U.S. player sets up near the 9-meter line while Canada players move into position.
- At roughly 0:06–0:08, the Canadian attacker takes a jump shot from the right side.
- The ball lands cleanly inside the court boundaries on the U.S. side without being touched by any defender.
- Because the ball clearly hits the floor within bounds and no touch occurs afterward, the point is awarded to Canada.

Verdict: Canada scores.

---

### qa_id=33485  ·  basketball / Descriptive
- **Q:** Does any team gain a point?
- **Gold:** `no`
- **sim:** 0.500 → **1.000**  (Δ +0.500)

  **BEFORE:** Yes, the Golden State Warriors win the game. The scoreboard indicates that at the start of the clip, the Warriors had 3 points while the San Antonio Spurs had 2. In the second half of the clip, the Warriors successfully defended against the Spurs' scoring attempt and scored their first point.

  **AFTER:**  Yes.

At roughly 0:04–0:06, the Miami player on the left side executes a quick jump assist that sends the ball into the net. The GSW defender dives but cannot reach it; the ball clearly crosses the goal line and hits the backboard area. This is a successful offensive kill.

Verdict: Miami gains a point.

---

### qa_id=89472  ·  fg / Descriptive
- **Q:** What is the video about?
- **Gold:** `Floor Exercise`
- **sim:** 0.500 → **1.000**  (Δ +0.500)

  **BEFORE:** The video features a gymnast named Leah Griesser performing a routine at what appears to be a competition venue. She starts by raising her arms, then jumps onto a mat, preparing for her next move. This jump suggests she might be attempting a trick or skill that requires significant height and flexibility. The setting includes banners advertising sponsors like "Shrin," indicating this could be part of a larger event organized by these organizations. As she prepares, there’s also mention of other athletes on the court, suggesting multiple participants in the competition. The environment looks pr

  **AFTER:**  This is a rhythmic gymnastics routine performed by Leah Griesser.

The clip begins with her standing at the start of the floor exercise, arms extended overhead. She immediately transitions into a dynamic movement sequence on the mat. Specifically, she executes a jump where she lands on one leg while extending the other backward, followed by a split position on the floor. This demonstrates flexibility and control typical of rhythmic gymnastics floor routines. The background shows competition setting details like banners and spectators, confirming this is a formal event.

---

### qa_id=31591  ·  football / Descriptive
- **Q:** Does any team gain a point?
- **Gold:** `no`
- **sim:** 0.500 → **1.000**  (Δ +0.500)

  **BEFORE:** Yes. The scoreboard shows Real Madrid at 1-0 ahead of Atlético Madrid, indicating they gained a point.

  **AFTER:**  The clip shows a soccer match between Real Madrid (in white) and Atletico Madrid (in blue/red). The scoreboard reads FM 0 - 1 ATM at the start of the clip.

At approximately 0:06–0:08, an Atlético player receives the ball near the penalty area and takes a free kick toward the goal. The Real Madrid goalkeeper dives but cannot stop the ball; it lands cleanly inside the goal on the backside.

This is a direct scoring action: the ball crossed the goal line and hit the net without being blocked or saved. In soccer rules, this awards one point to the attacking team.

Yes — Atletico Madrid gains a po

---

### qa_id=32275  ·  football / Descriptive
- **Q:** Is the shoot of the right team successful?
- **Gold:** `no`
- **sim:** 0.500 → **1.000**  (Δ +0.500)

  **BEFORE:** The ball crosses the net successfully.

  **AFTER:**  The clip shows a Liverpool (in red) player taking a free kick from outside the penalty area against Manchester City (in yellow). The Liverpool player strikes the ball with his foot near the top corner of the goal.

At approximately 0:04–0:06, we can observe the ball traveling over the crossbar and past the goalkeeper. It lands cleanly inside the goal on the Manchester City side. There is no touch by the goalkeeper or any other defender before the ball hits the ground.

Therefore, based on the visual outcome of the play: Yes, the shot was successful.

---

### qa_id=32331  ·  football / Counterfactual
- **Q:** Would the through pass of the right team be successful if the other team did not intercept?
- **Gold:** `no`
- **sim:** 0.500 → **1.000**  (Δ +0.500)

  **BEFORE:** The through pass of the right team would likely fail. The passing player is positioned too close to the opposing goalkeeper’s net. A quick interception attempt would result in a clear-cut loss, with the ball going straight into the goal without any significant clearance or bounce. This outcome demonstrates that the passing player’s positioning and timing were critical to ensuring the pass remained intact and safe for the receiving side.

  **AFTER:**  Yes.

At roughly 0:01–0:02, a Bayern player on the right side takes a long roll-forward pass toward the middle. The ball travels cleanly into the empty space near the center circle. At that exact moment, a M'gladbach defender dives and makes contact with the ball mid-air, redirecting it back into play.

If the defending team had not intercepted the ball, its trajectory would have carried it directly into the goal. The pass was well-placed and unblocked; there’s no indication of fouls or errors by the attacking team. Therefore, without the dive, the through pass results in a goal.

---

### qa_id=32323  ·  football / Descriptive
- **Q:** What is the video about?
- **Gold:** `football`
- **sim:** 0.500 → **1.000**  (Δ +0.500)

  **BEFORE:** A soccer player in a white jersey with green lettering and number 4 on his back kicks a soccer ball while being pursued by two players in black jerseys. The scene takes place at a stadium filled with spectators during a match between M'gladbach and FC Bayern Munich.

  **AFTER:**  The video captures a soccer match between M'gladbach and FC Bayern, currently tied at 3–3 in the 50th minute. The scene shows a wide court view during live play. A player in white (FC Bayern) receives the ball near the center circle and takes a running approach toward the goal. Another teammate in white runs alongside him while another player in black defends. The camera follows the offensive movement as the attacker circles the ball under pressure from defenders. No scoring event occurs within this clip; the focus is on positioning and defensive awareness typical of high-level football.

---

### qa_id=32533  ·  football / Counterfactual
- **Q:** Would the long pass of the left team be successful if the other team did not intercept?
- **Gold:** `no`
- **sim:** 0.500 → **1.000**  (Δ +0.500)

  **BEFORE:** The long pass of the left team was successfully intercepted by the right team. The interception occurred when the right team player jumped high and caught the ball mid-air, preventing it from reaching the opposing side. This indicates that the left team's pass was indeed successful and could have continued forward without intervention.

  **AFTER:**  Yes.

At roughly 0:06–0:08, the Japanese player on the right side executes a low, extended pass toward the middle-left corner of the USA court. The U.S. defender near the center dives and makes contact with the ball at the peak of the trajectory, redirecting it upward and away from the goal area.

If that defensive dive had not occurred, the ball’s path would have continued unimpeded into the open space behind the goal. In soccer rules, any clear lane leading directly into the goal without being blocked or deflected counts as a goal. Here, the angle and speed suggest the ball would have crosse

---

### qa_id=29533  ·  volleyball / Counterfactual
- **Q:** Would the right team success to do spike if the other team did not save the ball?
- **Gold:** `no`
- **sim:** 0.500 → **1.000**  (Δ +0.500)

  **BEFORE:** Yes. The right team would succeed to do the spike because they can force the ball into their own net without needing a return, and this is the only way to execute that specific technique.

  **AFTER:**  Yes.

At roughly 0:04–0:06, the Croatian player on the right side executes a powerful overhead spike toward the Canadian court. The ball travels sharply downward and lands cleanly inside the boundary lines near the front row, well past any defensive positioning. At that exact moment, the Canadian defenders are already moving but don’t make contact; their only reason the rally continues is because they attempt to dig the ball.

The spike’s trajectory and placement indicate it would have landed inbounds without the save. No net touch or block occurred during the attack phase—this was a direct of

---

### qa_id=29815  ·  volleyball / Counterfactual
- **Q:** Would the first spike of the right team be successful if the other team did not save the ball?
- **Gold:** `no`
- **sim:** 0.500 → **1.000**  (Δ +0.500)

  **BEFORE:** The first spike of the right team appears to be successful based on the visual evidence provided. The ball goes over the net successfully, indicating that the player executing the spike hit the ball past the net without it bouncing back into the opponent's court. This suggests that even without any defensive intervention from the opposing team, the offensive move was effective. Therefore, the first spike of the right team would indeed be successful.

  **AFTER:**  Yes.

At roughly 0:01–0:02, the player on the right side (in red) executes a powerful spike aimed at the backcourt corner near the sideline. The trajectory is clearly downward into the opponent’s court with no block or deflection visible. At that exact moment, a defender on the left dives and makes contact with the ball just before it hits the floor, preventing it from touching the ground.

The spike itself was unreturned—the ball was already descending sharply past the net and heading toward the floor. There is no indication the ball was out of bounds or hit the net; its path was direct and a

---

### qa_id=31064  ·  volleyball / Descriptive
- **Q:** What is the video about?
- **Gold:** `volleyball`
- **sim:** 0.500 → **1.000**  (Δ +0.500)

  **BEFORE:** The video shows a basketball game played on a court. The scoreboard indicates that France has scored 7 points while Australia has scored 4 points. A player in green is seen running towards the basket, preparing for a shot. As he shoots, another player jumps and spikes the ball into the hoop. The ball crosses the net successfully, indicating the successful shot.

  **AFTER:**  This is a women’s indoor volleyball match between France (FRA) and Australia (AUS), played at the FIVB World Championship in Rio 2016. The scoreboard shows France leading 7–4 in sets, with the current set tied at 2–2.

At 0.1 seconds, we see a French player (green jersey) preparing to receive serve — she’s crouched low, eyes on the ball, hands ready. At 0.4 seconds, the Australian server (blue jersey #18) has just completed her toss and is mid-air executing a powerful overhead spike toward the French side. The French blockers (#9 and another) are already moving into position but appear too lat

---

### qa_id=30146  ·  volleyball / Descriptive
- **Q:** Does any team score in the video?
- **Gold:** `no`
- **sim:** 0.500 → **1.000**  (Δ +0.500)

  **BEFORE:** Yes, the Russian team scores at the end of the clip.

  **AFTER:**  No, neither team scores in this video.

Here’s the breakdown:
- The clip shows a men’s indoor volleyball match at London 2012 between Bulgaria (BUL) and Russia (RUS).
- At the start, the scoreboard reads BUL 4–2 RUS.
- A Bulgarian player serves; the Russian side receives and sets up an attack.
- A Russian attacker spikes the ball over the net.
- The Bulgarian blockers attempt to defend but fail to stop the spike.
- The ball lands cleanly on the Bulgarian court near the backcourt, untouched by any defender.
- Since the spiked ball hit the court inside the boundaries without being returned, the 

---

### qa_id=30574  ·  volleyball / Counterfactual
- **Q:** Would the spike of the left team be successful if the other team did not save the ball?
- **Gold:** `no`
- **sim:** 0.500 → **1.000**  (Δ +0.500)

  **BEFORE:** The spike of the left team would likely fail to succeed because the right team has already set up a net to block it. The ball is positioned too far back on their side to hit the net without being intercepted. This indicates that the defense successfully blocked the opportunity to score the spike.

  **AFTER:**  Yes.

At roughly 0:01–0:02, the Brazilian attacker on the left side executes a powerful spike aimed at the back-right corner of the Russian court. The trajectory is steep and directed well inside the boundary lines; there’s no block or dig shown that redirects the ball. At the moment of contact, the ball is clearly above the net and descending toward the opponent’s court.

If the Russian defenders had not attempted to keep the ball in play—i.e., if they hadn’t made any defensive reaction—the ball would have landed cleanly inbounds on their side. No visual indication suggests the ball hit the n

---

### qa_id=93148  ·  fg / Temporal
- **Q:** What do the players perform after performing clear pike circle backward with 1 turn to handstand?
- **Gold:** `clear pike circle backward with 0.5 turn to handstand`
- **sim:** 0.329 → **0.811**  (Δ +0.482)

  **BEFORE:** The player performs a round-off.

  **AFTER:**  After executing the clear pike circle backward with one handstand, the player performs a **full cartwheel**.

Here’s the breakdown:

1.  **The Handstand:** The athlete begins in a handstand position, extending her legs straight back.
2.  **The Pike Circle Backward:** She rotates her torso and hips backward while keeping her legs extended in a pike shape. This is a dynamic skill often used in acrobatic gymnastics routines to demonstrate flexibility and control.
3.  **The Cartwheel:** Immediately following the rotation, she transitions into a full cartwheel. In this move, she pushes off the grou

---

### qa_id=92224  ·  fg / Temporal
- **Q:** What do the players perform after performing double salto backward tucked with 2 twist?
- **Gold:** `double salto backward tucked with 1 twist`
- **sim:** 0.341 → **0.805**  (Δ +0.463)

  **BEFORE:** The players perform a round-off into a jump.

  **AFTER:**  The clip shows a rhythmic gymnastics routine. At approximately 0:00–0:01, the athlete executes a double salto backward with a two-twist element (a full rotation while airborne, often used to demonstrate flexibility and control). Immediately following this, she lands and transitions into a seated position on the mat, lowering her body until her legs are extended straight out in front of her torso — this is known as a "seated straddle" or "barbell sit" variation depending on leg positioning; here, it appears to be a controlled seated position with legs fully extended horizontally.

After complet

---

### qa_id=28676  ·  aerobic_gymnastics / Temporal
- **Q:** What do the players perform after performing explosive push up?
- **Gold:** `aerobic gymnastics illusion`
- **sim:** 0.370 → **0.815**  (Δ +0.444)

  **BEFORE:** The players perform a split jump.

  **AFTER:**  After executing an explosive push-up position (where the hands touch the floor and the legs extend straight back), the athletes immediately transition into a **split leap**.

In this move, they jump vertically while extending their legs fully apart—one leg forward and one backward—forming a horizontal split in mid-air. This is a standard element in aerobic gymnastics routines, demonstrating flexibility, power, and synchronization. The sequence flows directly from the strength of the push-up into the dynamic height of the leap.

---

### qa_id=27743  ·  aerobic_gymnastics / Temporal
- **Q:** What do the players perform after performing explosive push up?
- **Gold:** `aerobic gymnastics support`
- **sim:** 0.423 → **0.846**  (Δ +0.423)

  **BEFORE:** After performing an explosive push-up, the players transition into a dynamic sequence involving a jump, a spin, and a subsequent backhand kick. This indicates they move from their standing position into active athletic movement, likely preparing for a high-intensity activity or drill. The sequence suggests a progression of movements where one action sets the stage for another more complex or rapid motion.

  **AFTER:**  After executing the explosive push-ups, the two female aerobic gymnasts transition into a synchronized **split leap**.

Here’s the breakdown:

- At approximately 0:01–0:02, both athletes complete their explosive push-ups and immediately launch into the air.
- Their bodies extend fully into a horizontal split position mid-air — one leg forward, one back, torso upright.
- They land cleanly on both feet and hold the pose momentarily before moving into the next sequence.

This is a classic element in aerobic gymnastics routines, often used to demonstrate flexibility, control, and synchronization. 

---

### qa_id=28997  ·  aerobic_gymnastics / Temporal
- **Q:** What do the players perform before performing split?
- **Gold:** `aerobic gymnastics support`
- **sim:** 0.423 → **0.808**  (Δ +0.385)

  **BEFORE:** In the clip, the players first perform a push-up before executing a split jump. This sequence demonstrates a clear progression of athletic movements leading up to the final move.

  **AFTER:**  The clip shows four female rhythmic/aerobic gymnasts on a floor exercise mat during a synchronized routine. At approximately 0:00–0:01, they begin in a low crouch position, hands placed firmly on the floor, knees bent deeply, and bodies leaning forward — this is a preparatory stance often used to generate momentum or transition into dynamic movements.

Immediately after this crouched position, at roughly 0:01–0:02, all four athletes simultaneously execute a full split leap (also known as a straddle jump or split jump) while airborne, legs extended fully apart horizontally.

Therefore, the acti

---

### qa_id=28592  ·  aerobic_gymnastics / Temporal
- **Q:** What action do the players perform before helicopter?
- **Gold:** `aerobic gymnastics straddle jump`
- **sim:** 0.438 → **0.812**  (Δ +0.375)

  **BEFORE:** The players perform a split jump before the helicopter appears.

  **AFTER:**  The clip shows three female rhythmic gymnasts performing on a mat. At approximately 0:01–0:02, all three execute a synchronized leap where they extend their legs into a straddle position while airborne. This is known as a "straddle jump" or "split leap." Immediately following this jump, they transition into a handstand-like position, which resembles the motion of a helicopter spin (a dynamic rotational movement often used in rhythmic gymnastics). The straddle jump serves as the preparatory element directly preceding the helicopter turn.

Answer: Straddle jump.

---

### qa_id=32827  ·  football / Descriptive
- **Q:** What are the players doing in the video?
- **Gold:** `football`
- **sim:** 0.625 → **1.000**  (Δ +0.375)

  **BEFORE:** The players are playing soccer on a grass field during a match. They are running around, passing the ball, and attempting to score. The scoreboard shows that Mexico is leading Germany 2-0.

  **AFTER:**  This is a men’s international friendly match between Germany (GER) and Mexico (MEX), currently tied at 0–0 in the second half (22:38). The camera angle is wide, showing both teams spread across the field.

At approximately 0.4 seconds into the clip, we see a German player (in white jersey #19) receiving a pass near the center circle. He immediately turns toward the goal and takes a jump shot with his right foot. His body is fully extended, aiming straight at the goal mouth.

The Mexican goalkeeper (in yellow) dives but fails to reach the ball — it lands cleanly inside the goal. No other defend

---

### qa_id=29632  ·  volleyball / Descriptive
- **Q:** How does the left team gain a score in the video?
- **Gold:** `volleyball spike`
- **sim:** 0.500 → **0.875**  (Δ +0.375)

  **BEFORE:** The left team gains a score by successfully scoring a basket. The visual evidence shows the ball being released from the net, which is a key indicator that a successful shot has occurred. This action directly contributes to the team's score.

  **AFTER:**  The left team (in red) scores via a successful attack.

At the start of the clip, the right team (in blue) sets up an offense. A player on the right side jumps at the net and executes a powerful spike toward the left court. The left team’s players attempt to block but fail; one defender dives but cannot reach the ball. The spiked ball lands cleanly inside the boundary lines on the left team’s side of the court, resulting in a direct point for the attacking team.

This is a classic offensive kill in volleyball — specifically, a hard-driven spike that evades all defensive coverage. No scoreboard

---

### qa_id=28299  ·  aerobic_gymnastics / Temporal
- **Q:** What do the players perform before performing straight jump?
- **Gold:** `aerobic gymnastics illusion`
- **sim:** 0.444 → **0.815**  (Δ +0.370)

  **BEFORE:** Before performing the straight jump, the players execute a series of warm-up exercises including a push-up, a split, and a round-off. The sequence begins with these preparatory movements which set up the subsequent jump.

  **AFTER:**  In the clip, we observe a group of female rhythmic/aerobic gymnasts performing on a stage at the "5th Aerobic Gymnastics Asian Championships." The sequence begins with the athletes moving into position.

At approximately 0:00–0:01, all four athletes simultaneously execute a **round-off**, which is a cartwheel-like entry where they land on their hands and feet while rotating sideways. This is followed immediately by a **straight jump**, where they leap vertically with legs extended.

The key action performed directly before the straight jump is the **round-off**. It serves as the preparatory mo

---

### qa_id=34443  ·  basketball / Descriptive
- **Q:** Does the left team do 2-point shot successfully?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** No, the left team does not do 2-point shot successfully. The left team’s player attempts a layup but misses, resulting in a missed shot that ends the offensive possession.

  **AFTER:**  Yes.

At roughly 0:06–0:08, the player on the left side (white jersey #5) approaches from the left wing, jumps near the net, and executes a powerful overhead hit toward the right side of the court. The defender on the right dives but cannot reach it; the ball lands cleanly inside the boundary lines on the opponent’s side without being touched afterward. This is a completed 2-point shot.

---

### qa_id=34642  ·  basketball / Descriptive
- **Q:** Does any team score in the video?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** No. The scoreboard shows IRI leading JPN 4–6 in the first quarter with seven minutes remaining. Both teams are still on offense; neither has taken a free throw yet. There is no visual indication that either team scored during this clip.

  **AFTER:**  No, neither team scores in this video clip.

The sequence shows a rally during the World Cup Qualifier match between Iran (IRI) and Japan (JPN). At the start of the clip, the scoreboard reads IRI 4–6 JPN. The Japanese player in red executes a jump serve that travels over the net into Iran’s court. An Iranian defender attempts to block but fails; the ball lands cleanly on the Iranian side near the backcourt without being touched again. Since the ball touches the floor inside the boundary lines on the opponent’s side, the point is awarded to Japan.

However, the question asks whether *any* team 

---

### qa_id=33909  ·  basketball / Descriptive
- **Q:** Is the 2-point shot of the left team successful?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** No. The ball went into the backboard, out of bounds, and the left team scored 4 points on this turn. This indicates the 2-point attempt failed.

  **AFTER:**  The clip shows a women’s Olympic beach volleyball match at Rio 2016. The left team (in white) has possession near the backcourt. At roughly 0:04–0:05, the player on the left side jumps and attempts a powerful overhead hit toward the opponent’s court.

At 0:06, the ball is clearly seen descending sharply into the opponent’s court boundaries — specifically, just inside the sideline near the net area. It bounces once on the floor before rolling out of bounds. No touch occurs after the shot; the ball does not cross the net or deflect off any defender.

This is a direct kill: the attacker successfu

---

### qa_id=33604  ·  basketball / Descriptive
- **Q:** Does any team score in the video?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** No. The scoreboard shows ANG 33, PHI 26, indicating this is a match where neither team has scored yet. There are no shots taken, no baskets hit, and no players scoring. The only event shown is a player celebrating after making a move, but there is no indication that they scored anything. Therefore, no team scores in the video.

  **AFTER:**  Yes, the team in white jerseys scores a point.

At approximately 0:01–0:02, a player in the white jersey (#14) executes a jump serve over the net into the opposing court. The players on the blue side react—some turn away, others prepare to receive—but the ball lands cleanly inside their court boundaries without being touched again. This constitutes a successful serve that results in a point for the serving team. No scoreboard updates are needed; the visual confirmation of the ball hitting the floor confirms the point was scored.

---

### qa_id=33896  ·  basketball / Descriptive
- **Q:** Does the left team do 2-point shot successfully?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** The left team does not score on the 2-point shot. The ball crosses the net without hitting the backboard, which indicates the shot failed.

  **AFTER:**  The clip shows a men’s Olympic beach volleyball match at London 2012. At roughly 0:04–0:06, a player on the right side (in blue) executes a powerful overhead spike toward the left side of the court. The ball travels fast and low, well inside the boundary lines near the sideline. Two defenders on the left attempt to block but fail; one dives but cannot reach it. The ball lands cleanly inbounds without being touched by any other players afterward.

This is a classic “2-point shot” in beach volleyball — a direct attack that scores if unreturned. Since the ball clearly lands in bounds and no touch

---

### qa_id=33494  ·  basketball / Descriptive
- **Q:** Does the right team do 2-point shot successfully?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** No, the right team does not do 2-point shot successfully. The ball crosses the net without touching the rim.

  **AFTER:**  The clip shows a men’s NBA game between USA (in white) and AUS (in green). The camera follows a play near the basket. At approximately 0:08–0:09, the Australian player on the right side of the court executes a jump shot toward the left side of the court.

Observing the trajectory and impact: the ball clearly passes through the net of the hoop without touching any American defenders. The ball lands cleanly inside the boundary lines on the USA side, well beyond any defender’s reach. There is no block, deflection, or touch that would indicate a successful save.

Verdict: Yes, the right team (Aust

---

### qa_id=88814  ·  fg / Temporal
- **Q:** Do the players perform stretched hop or jump with 1 turn before the aerial cartwheel?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** The players perform a stretch hop followed by a jump before the aerial cartwheel. The first move involves stepping side-to-side on the mat, which appears to be a stretch hop. This is immediately followed by a jump that includes turning once. Finally, they execute the aerial cartwheel.

  **AFTER:**  The clip shows a rhythmic gymnastics routine at the 2016 American Cup. At approximately 0:01–0:02, the athlete executes a turn while airborne on one leg, which is characteristic of an aerial cartwheel. The movement is clearly a single rotational turn followed by the subsequent aerial cartwheel. There is no indication of a stretched hop or jump preceding this sequence. Therefore, the correct answer is:

Yes, the player performs a stretch hop/jump with 1 turn before the aerial cartwheel.

---

### qa_id=90085  ·  fg / Temporal
- **Q:** Do the players perform (swing backward) with 0.5 turn and straddle flight backward over high bar to catch high bar before the johnson (switch leap with 0.25 turn to side split or to straddle pike position)?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** No. The player performs a swing backward with a 0.5 turn, but does not straddle flight backward over high bar to catch the high bar. Instead, she executes a jump forward and lands on her feet.

  **AFTER:**  The clip shows a rhythmic gymnastics routine. At approximately 0.2–0.4 seconds into the clip, the athlete on the left performs a **straddle flight backward** over the high bar while making a **0.5 turn**. Immediately following this, she transitions into a **switch leap with a side split**, which occurs at roughly 0.6 seconds.

Regarding the "johnson" mentioned in the question — this appears to be a misinterpretation. In rhythmic gymnastics, there is no such named element as a "johnson." However, if we assume the user meant **"switch leap"** (which matches the visual sequence), then the analysi

---

### qa_id=88077  ·  fg / Temporal
- **Q:** Do the players perform double salto backward tucked with 1 twist before the tuck hop or jump with 1 turn?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** No. The player performs a double salto backward with one twist, followed immediately by a tuck hop/jump with one turn. This sequence occurs right at the beginning of the clip, prior to any other moves involving turns.

  **AFTER:**  Yes.

At roughly 0:01–0:02, the female athlete executes a classic back handspring-style entry: she plants her feet, kicks up into a full twist (one twist), and lands on both feet facing forward again. Immediately after that, at 0:03–0:04, she transitions directly into a tuck hop/jump with one turn. The sequence is clear: back handspring → tuck hop/jump with one turn. No other intermediate moves occur between these two elements.

---

### qa_id=31279  ·  football / Counterfactual
- **Q:** Would the long pass of the left team be successful if the other team did not intercept?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** The provided clip shows a soccer match in progress during London 2012. The scoreboard indicates that POK (Prague Olympians) has scored against COL (Colombia), with POK leading 1-0. The long pass of the left team appears to be part of this ongoing play. However, there is no visual information available to determine if this particular long pass was successful under the condition that the other team did not intercept. Therefore, I cannot provide a definitive answer based on the visual evidence present in the clip.

  **AFTER:**  Yes.

What I see: At roughly 0:01–0:02, the player on the left side (yellow jersey) receives the ball near the sideline and executes a very high, low arc pass toward the center of the court. The trajectory is steep and aimed into the open space between the two midfield players.

Evidence: The pass’s path clearly avoids the defending block and lands inside the opponent’s goal area without being touched by any defender. No contact occurs after the pass; the ball travels unimpeded toward the goal.

Verdict: Without the intercept, this long pass would have resulted in a direct goal for the left te

---

### qa_id=33172  ·  football / Counterfactual
- **Q:** Would the left team success to do long pass if the other team did not intercept?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** The left team would fail to complete the long pass successfully because they were intercepted by the opposing player during the run-up to the shot. The interception occurred while the left team was attempting to advance the ball towards the basket, which prevented any further progress on that side of the field. This indicates that the interception disrupted the intended trajectory of the pass.

  **AFTER:**  Yes.

At roughly 0:01–0:02, the Brazilian player on the left side (in yellow) receives the ball near the sideline and immediately pivots into a low, controlled approach toward the goal. The French goalkeeper dives but cannot reach it; the ball lands cleanly inside the goal.

Evidence:
- The Brazilian player makes clean contact with the ball at the touchpoint.
- His follow-through is complete and directed straight into the goal.
- No other defender touches the ball during this sequence.

Counterfactual: If the French goalkeeper had not intervened, the long pass would have resulted in a goal.

---

### qa_id=31584  ·  football / Descriptive
- **Q:** Is the steal of the right team successful?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** No. The player who attempted the steal fell to the ground and did not take possession of the ball. This indicates that the steal attempt was unsuccessful.

  **AFTER:**  The clip shows a soccer match between Bahrain (BHA) and Morocco (MU). At approximately 0:01–0:02, a player in a red jersey (Morocco) steals the ball near the goal area. The other players immediately attempt to defend; one player dives but cannot stop the ball from entering the goal. The ball clearly lands inside the goal mouth at 0:03, with no rebound or deflection that would indicate a save. Therefore, the steal by the right team (Morocco) is successful. Yes.

---

### qa_id=32875  ·  football / Counterfactual
- **Q:** Would the right team success to do through pass if the other team did not intercept?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** No, the right team could not successfully do through pass if the other team did not intercept.

  **AFTER:**  Yes.

What I see: At roughly 0–1 s into the clip, Dortmund’s left-side player (in yellow) receives the ball near the center circle and immediately passes it forward toward his teammate on the far wing. The passing lane is open; no defender is positioned to block or deflect it.

Why it succeeds without the intercept: The pass is clean and directed into a free zone. There’s no immediate threat of a corner or tip that would force a save. So, absent any interception, the pass results in a direct scoring opportunity.

---

### qa_id=32964  ·  football / Counterfactual
- **Q:** Would the through pass of the left team be successful if the other team did not intercept?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** No, because there is no evidence that the opposing team intercepted the ball during this clip. The only information available is that the ball passed through the net without being blocked or contested. Therefore, we cannot conclude whether the interception occurred based solely on the visual data provided.

  **AFTER:**  Yes.

At roughly 0:01–0:02, the player on the left side (in white) executes a quick overhead pass toward the center of the court. The teammate who receives it immediately moves into position and prepares to drive forward. In that frame, the passing lane is open and the receiving player is already moving laterally to cover the goal area.

If the opposing team had not intercepted this pass—i.e., if the defender on the right side had not made contact—the ball would have continued its trajectory into the opponent’s goal. There is no visual indication that the pass was blocked or deflected; it appe

---

### qa_id=32233  ·  football / Counterfactual
- **Q:** Would the first long pass of the right team be successful if the other team did not intercept?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** No. The pass was intercepted by the defending team during the second half of the clip.

  **AFTER:**  Yes.

At roughly 0:06–0:08, the Brazilian player on the right side receives a corner kick and makes a low pass toward the center. The French defender dives but cannot reach it; the ball clearly lands inbounds near the baseline without being touched again. That is a direct scoring opportunity. Since the pass was well-placed and the defender was out of position, the first long pass by the right-side team would have been successful had there not been an interception.

---

### qa_id=32443  ·  football / Counterfactual
- **Q:** Would the right team success to do through pass if the other team did not intercept?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** The right team would succeed to do through pass if the other team did not intercept. The right team has possession of the ball and can execute the pass without obstruction. If the opposing team blocks the pass, their interception attempt will fail, allowing the right team to complete the through pass successfully.

  **AFTER:**  Yes.

At roughly 0:04–0:06, a Japanese player in red (#10) receives the American serve near the 9-meter line and immediately pivots into a set position. At 0:07 he executes a quick turn and passes the ball cleanly toward his teammate on the left side of the court. The trajectory is low and well-placed; there’s no block or deflection shown that would prevent the pass.

Evidence:
- Clean footwork and body positioning indicate proper approach and intent.
- The ball is passed directly into the open space ahead without being touched by any defender.
- No visual indication of a dig or save.

Counter

---

### qa_id=32242  ·  football / Descriptive
- **Q:** Does the left team do through pass successfully?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** No, the left team does not make a successful through pass. The ball remains in possession of the right team throughout this segment of gameplay.

  **AFTER:**  Yes.

At roughly 0:01–0:02, the Chelsea player on the left side receives the Liverpool’s approach and executes a clear through pass toward the center near the archway. The ball travels smoothly over the goalkeeper’s outstretched hands and lands cleanly inside the goal area without being touched again. That constitutes a successful through pass.

---

### qa_id=31312  ·  football / Counterfactual
- **Q:** Would the left team success to do cross if the other team did not intercept?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** No, because there is no visual evidence showing that the left team could successfully execute a cross without being intercepted by the opposing team. The clip shows players running towards each other, but no clear indication of a successful pass attempt or interception occurred. Therefore, we cannot definitively conclude that a cross could succeed without interference based on this clip alone.

  **AFTER:**  Yes.

At roughly 0:00–0:01, a player on the left side (in red) receives a pass near the center-right of their half and immediately pivots into a low squat position while keeping his legs straight. This is a classic “cross”—a lateral crossbody-like movement used to redirect a teammate’s path without dropping to the ground. The key is that he initiates the turn cleanly, maintains control of his hips, and keeps his body aligned with the trajectory of the passing lane.

The right-side defender (white jersey) is already moving laterally but does not reach the ball until it’s too far out of bounds. 

---

### qa_id=31245  ·  football / Counterfactual
- **Q:** Would the third cross of the left team be successful if the other team did not intercept?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** The third cross of the left team would likely be successful. In the provided clip, the left team player performs a cross at approximately 47.76 seconds. The opposing team's goalkeeper, who is positioned near the center of the goal, does not appear to have intercepted this particular cross attempt. This lack of interception suggests that the ball traveled into the net, indicating a successful third cross for the left team.

  **AFTER:**  Yes.

At roughly 47:26–47:28, a Liverpool player in blue (#10) receives a pass near the edge of the Chelsea penalty area and takes a low, powerful shot toward the far corner of the net. The Chelsea goalkeeper dives and makes contact with the ball just as it crosses the goal line; his dive redirects the ball back into Liverpool’s court.

If the Chelsea players had not made that dive, the trajectory of the shot would have carried it cleanly past the goal. There is no indication of a block or deflection that would have altered its path. Therefore, the third cross by the left team (Liverpool) woul

---

### qa_id=31957  ·  football / Descriptive
- **Q:** Is the steal of the left team successful?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** No. The player who attempted a left-side tackle fails to get the ball into play; instead, he is tackled by the defending team’s goalkeeper. This indicates that the steal attempt was unsuccessful.

  **AFTER:**  The clip shows a soccer match between LUTV (in yellow) and WBA (in white). At approximately 0:04–0:06, a player on the left side (LUTV) receives the ball near the penalty area and takes a run toward the goal. A defender from WBA jumps at the net but fails to stop the ball. The ball clearly lands inside the goal line on the right side of the goal (from our view), which means the goal was scored.

Yes, the steal was successful. The attacker kicked the ball into the goal without being blocked or deflected out. No scoreboard or ref signal is needed—the visual result of the ball hitting the goal co

---

### qa_id=32399  ·  football / Descriptive
- **Q:** Is the long pass of the right team successful?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** No. The right team’s long pass is unsuccessful because the ball does not cross the goal line when it should have.

  **AFTER:**  The clip shows a soccer match between USA (in red) and NED (in white/blue). At approximately 74:38–74:40, a U.S. player receives a pass near the center and initiates a run toward the goal. The Netherlands defender on the left side blocks his path.

At 74:41, the U.S. attacker executes a very high, extended aerial pass (a “long pass” or “flare”) that arcs sharply over the block and lands cleanly inside the goal area, just past the goal line. There is no touch by any other player who could have altered its trajectory.

Verdict: Yes, the long pass was successful. It resulted in a direct goal for 

---

### qa_id=31637  ·  football / Descriptive
- **Q:** Does any team score in the video?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** No. The scoreboard shows CHE leading ARS 1-0 at 19:35, indicating no points were scored yet.

  **AFTER:**  Yes.

At roughly 0:02–0:03, an Arsenal player (in red) receives a pass near the edge of Chelsea’s penalty area and takes a jump shot toward the goal. The Chelsea goalkeeper dives but cannot stop the ball; it hits the net at about 0:03–0:04. Since the ball crossed the goal line and entered the net, that constitutes a goal for Arsenal. No scoreboard change is needed—the visual result of the shot being inbounds confirms the point.

---

### qa_id=31816  ·  football / Counterfactual
- **Q:** Would the long pass of the right team be successful if the other team did not intercept?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** No, the long pass would likely fail because the opposing team intercepted it during its trajectory. The interception occurred just as the ball was approaching the opponent's defensive zone, preventing any potential scoring opportunity for the attacking team.

  **AFTER:**  Yes.

At roughly 0:49–0:50, the right-side team (in white) receives the ball near the center circle and executes a long pass toward the goal. The left side’s goalkeeper dives but cannot reach it; the ball crosses the goal line without being touched again. That is a direct goal. So, absent any interception, the pass would have been successful.

---

### qa_id=32356  ·  football / Counterfactual
- **Q:** Would the left team success to do through pass if the other team did not intercept?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** No. The right team intercepted the ball during the first second of the clip, preventing any further passing attempts by the left team. Therefore, the left team could not have successfully passed to score through that interception.

  **AFTER:**  Yes.

What we see: At roughly 0:01–0:02, a Thai player (light blue jersey, #8) receives a pass near the center and immediately turns toward the goal while keeping her legs straight and arms ready. Her teammate (#10) is already moving into position behind her. The Australian defender on the right corner has just made contact with the ball at the net and is attempting to block/defend.

Reasoning:
- The Thai player’s trajectory and positioning indicate she is prepared to continue the offensive run under pressure.
- The interception attempt by Australia is imminent; there’s no visual cue that the 

---

### qa_id=32060  ·  football / Descriptive
- **Q:** Is the through pass of the left team successful?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** No. The right side of the screen shows the right team receiving possession, indicating that the left team’s through pass was unsuccessful.

  **AFTER:**  The clip shows a soccer match between Tottenham (in red) and Arsenal (in white). At approximately 0:08–0:09, an Arsenal player receives a pass near the center circle and immediately drives toward the goal. The Tottenham goalkeeper dives but cannot reach the ball; it lands cleanly inside the goal mouth at 0:09–0:10.

This is a classic “through pass” — a free kick or challenge shot that bypasses the goalkeeper’s defensive reach and lands inbounds. Since the ball clearly entered the goal without being touched by any defender, the pass resulted in a direct goal.

Verdict: Yes, the through pass was

---

### qa_id=31403  ·  football / Counterfactual
- **Q:** Would the left team success to do long pass if the other team did not intercept?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** The left team failed to execute a long pass. The right team intercepted the ball at approximately the 1-second mark. This interception prevented the left team from continuing their possession, resulting in a loss of possession. Therefore, the left team could not have successfully completed the pass without the right team's intervention.

  **AFTER:**  Yes.

At roughly 0:00–0:01, the Japanese player in white (#9) receives the ball near the center and immediately pivots into a low, wide position—essentially a forearm pass (bump) aimed toward the far sideline. The French defender on the right side dives but doesn’t make contact; the ball travels cleanly past him into the empty space behind the goal area.

If the opposing team had not intercepted the ball (i.e., if the dive hadn’t occurred), the long pass would have landed untouched inbounds on the opponent’s side. There is no visual indication of a block or tip that altered its trajectory befo

---

### qa_id=31385  ·  football / Descriptive
- **Q:** Does the right team do steal successfully?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** No. The right team does not steal successfully. They attempt to gain possession but fail to secure control due to the left team’s defensive effort.

  **AFTER:**  The clip shows a men’s soccer match between Brazil (BRA) and Belgium (BEL). At approximately 0:01–0:02, a Brazilian player on the left side of the court receives a pass near the 9-meter line and immediately drops into a low crouch while extending his arms forward — this is a classic “steal” (also called a “roll”) used to bypass defensive pressure. The Belgian defenders are already moving toward him; one attempts to block but fails to stop the ball. By 0:03, the ball has clearly crossed the goal line and landed inside the net — visible as the red-clad goalkeeper dives but cannot reach it.

This

---

### qa_id=32339  ·  football / Counterfactual
- **Q:** Would the left team success to do cross if the other team did not intercept?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** No. The clip shows the left team attempting a cross, but the right team intercepted the ball before it could reach the opponent’s side. This interception prevented the left team from scoring that specific pass. Therefore, even without an actual interceptive player on the field, the act of a cross being attempted does not guarantee success; the presence of an opposing defender who can stop it determines the outcome.

  **AFTER:**  Yes.

At roughly 0:04–0:06, the player on the left side (in red) receives a pass near the penalty area and takes a jump approach toward the goal. The goalkeeper dives but does not touch the ball; the ball crosses the goal line into the net. Since the attacker’s trajectory and timing show he would have completed the cross had there been no defensive intervention, the action succeeds.

---

### qa_id=29958  ·  volleyball / Counterfactual
- **Q:** Would the spike of the left team be successful if the other team did not save the ball?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** No, because the opponent saved the ball. The ball remained in play during the spike.

  **AFTER:**  Yes.

At roughly 0:00–0:01, Brazil’s outside hitter on the left side executes a powerful spike toward the U.S. right-side zone. The trajectory is steep and aimed well inside the court boundaries; there’s no block touching the ball at the net. The only reason the rally continues is that a U.S. defender dives and makes contact with the ball just before it hits the floor. Without that defensive touch, the ball would have landed cleanly inbounds on the opponent’s side—no need for a save. So, absent the save, the spike results in a direct point for Brazil.

---

### qa_id=29957  ·  volleyball / Counterfactual
- **Q:** Would the left team success to do spike if the other team did not save the ball?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** No, because the opponent saved the ball. The ball landed on the ground in front of their player, preventing him from executing the spike.

  **AFTER:**  Yes.

At roughly 0:01–0:02, Brazil’s middle blocker on the left side executes a jump set near the net and spikes the ball sharply downward into the USA court. The trajectory is steep and aimed at the back-right corner of the opponent’s side. At that exact moment, two U.S. defenders are already moving laterally; one dives to his right and makes contact with the ball just before it hits the floor, preventing the spike from touching the ground.

The key is timing and positioning: the Brazilian player had enough time to reach the ball mid-air, and the U.S. defense was already reacting to its path.

---

### qa_id=30610  ·  volleyball / Counterfactual
- **Q:** Would the spike of the left team be successful if the other team did not save the ball?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** The spike of the left team would likely succeed, but this depends on the defensive response. The right team’s defense appears to be positioned well enough to stop the shot, making it difficult to predict a successful outcome without seeing the actual ball impact. Therefore, the most accurate assessment is that the spike could have been successful had the defense reacted quickly and effectively.

  **AFTER:**  Yes.

At roughly 0:04–0:06, the Canadian player on the left side executes a powerful spike aimed at the back-right corner of the Croatian court. The trajectory is steep and well-placed; the only reason the rally continues is that two Croatian defenders dive and make contact with the ball just before it hits the floor. Without that save, the ball would have struck the playing surface cleanly inside the boundary lines—no net touch, no block deflection, and no out-of-bounds angle. So, absent the dig, the spike results in a direct point for Canada.

---

### qa_id=30447  ·  volleyball / Counterfactual
- **Q:** Would the first spike of the left team be successful if the other team did not save the ball?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** No. The first spike of the left team was unsuccessful because the opposing team saved it, preventing the ball from reaching the net. This demonstrates that saving the ball is a critical factor in determining the outcome of such a shot.

  **AFTER:**  Yes.

At roughly 0:01–0:02, the Turkish player on the right side executes a powerful spike toward the Russian court’s backcourt near the sideline. The ball is driven sharply downward and well inside the boundary lines; there is no block contact at the net that deflects it. The only reason the rally continues is because a Russian defender dives and makes contact with the ball just before it hits the floor.

If the dive had not occurred, the trajectory shows the ball clearly hitting the court surface within bounds—no touch from any other player. Therefore, without the save, the spike would have 

---

### qa_id=30429  ·  volleyball / Descriptive
- **Q:** Is the defend of the left team successful?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** No. The defending team failed to successfully block the netting during this volley.

  **AFTER:**  Yes.

At roughly 0:01–0:02, the Turkish player on the right side executes a powerful spike down the line. The Russian defender on the left dives forward but cannot reach the ball; it lands cleanly inside her court near the sideline. Since the ball touched the playing surface without being returned, the defense failed.

---

### qa_id=29910  ·  volleyball / Counterfactual
- **Q:** Would the spike of the right team be successful if the other team did not save the ball?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** No. The spike was unsuccessful because the ball bounced back into the net, indicating that the opposing player caught it or deflected it away. This demonstrates the importance of accurate defense and ball control during the game.

  **AFTER:**  Yes.

At roughly 0:07–0:09, the Japanese player on the right side executes a powerful overhead spike aimed at the back corner of Peru’s court. The trajectory is steep and well-placed; there’s no block deflecting the ball. At that exact moment, a Peruvian defender dives and makes contact with the ball just before it hits the floor, preventing it from touching the court.

If the dive had not occurred—i.e., if the ball had landed untouched on the court—the spike would have resulted in a direct point for Japan. The angle and speed indicate the ball was likely to bounce cleanly inside the boundary 

---

### qa_id=30727  ·  volleyball / Counterfactual
- **Q:** Would the first spike of the left team be successful if the other team did not save the ball?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** The first spike of the left team would likely fail. The opponent has already made a save attempt at that moment, preventing the shot from going into the hoop. Therefore, even without saving the ball, the shot would not succeed.

  **AFTER:**  Yes.

At roughly 0:08–0:09, the Chinese player on the left side executes a powerful spike from the front row. The trajectory is steep and aimed well inside the opponent’s court near the sideline. At that exact moment, the Serbian defender dives and makes contact with the ball just before it hits the floor, preventing it from touching the ground.

If the dive had not occurred—i.e., if the ball had landed cleanly on the court—the spike would have resulted in a direct point for China. There is no visual indication of the ball going out of bounds or being blocked; its path clearly intersects the c

---

### qa_id=30730  ·  volleyball / Counterfactual
- **Q:** Would the left team success to do their second spike if the other team did not save the ball?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** No. The left team did not succeed in doing their second spike because the right team saved the ball. The right team successfully defended the spike, preventing the left team from scoring on that attempt. This demonstrates that defense is crucial; even if a player attempts a risky shot, a successful defensive block can negate the opportunity to capitalize on the shot.

  **AFTER:**  Yes.

At roughly 0:07–0:08, the Chinese player on the back row executes a powerful cross-court spike aimed at the corner near the sideline. The Serbian defenders attempt to dig the ball but fail to keep it in play; the ball lands cleanly inside the court boundary without touching any other defender. Since the trajectory and placement were direct threats into open space, and the only reason the rally continued was the successful dig, the spike would have resulted in a point had the defense not saved it.

---

### qa_id=30030  ·  volleyball / Counterfactual
- **Q:** Would the second attack of the right team be successful if the other team did not save the ball?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** No, the second attack of the right team would not have been successful because the opposing team saved the ball. The visual evidence shows that the defending team intercepted the ball during the first attempt and returned it to their own side. This interception prevented the scoring opportunity from occurring.

  **AFTER:**  Yes.

At roughly 0:01–0:02, the Indian player on the right side executes a powerful spike that lands cleanly inside the Korean court near the endline. The Korean players attempt a dig but fail to keep the ball in play; it bounces once and rolls out of bounds without being touched again. There is no visual indication of a net touch or error during the spike itself—only the unforced drop into the opponent’s territory. Therefore, absent the defensive save, the attack would have resulted in a point for India.

---

### qa_id=30029  ·  volleyball / Counterfactual
- **Q:** Would the left team success to do spike if the other team did not save the ball?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** No. The ball was saved during the spike attempt on the right side.

  **AFTER:**  Yes.

At roughly 0:01–0:02, the Indian player on the right side executes a powerful spike from the front row. The trajectory is steep and aimed at the backcourt corner near the sideline. At that instant, two KOR defenders are already positioned under the net; one dives low and makes contact with the ball just before it hits the floor.

If we remove that defensive touch, the ball would have landed cleanly inside the boundary lines on the opponent’s court. There is no indication the spike was blocked or tipped out-of-bounds. Therefore, the spike would have resulted in a point for the left team (

---

### qa_id=29628  ·  volleyball / Descriptive
- **Q:** Does any team gain a point?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** No. The clip shows a rally where both teams attempt shots at each other’s end lines, but neither team scores a point. No points are awarded, and the scoreboard remains unchanged throughout the sequence.

  **AFTER:**  Yes.

At roughly 0:08–0:10, the German player in red (#14) executes a powerful spike that lands cleanly on the U.S. side near the baseline, untouched by defenders. The ball clearly touches the court surface inside the boundary lines, which under FIVB rules awards one point to the attacking team. No scoreboard change is needed—the visual outcome confirms the point.

---

### qa_id=29641  ·  volleyball / Descriptive
- **Q:** Is the protect of the right team successful?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** The player on the left side of the screen successfully protects the right side of the net, preventing the opposing player from scoring. The ball remains in his hands throughout the sequence. Therefore, the protect of the right team is successful.

  **AFTER:**  The clip shows a men’s Olympic beach volleyball match at London 2012 between USA (in red) and Germany (in blue). The camera angle is wide, showing both teams on opposite sides of the net.

At approximately 0.7–0.8 seconds into the clip, the German player on the right side (wearing #14) dives forward near the sideline to attempt a dig — this is the “protect” action referenced in the question. He makes contact with the ball just before it hits the court.

Observing the trajectory: the ball clearly lands cleanly inside the boundary lines on the German side, untouched by any other player. There is

---

### qa_id=30849  ·  volleyball / Descriptive
- **Q:** Does the right team do spike successfully?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** The right team does not succeed in doing a spike. The ball goes into the net, which is why the spike fails.

  **AFTER:**  The clip shows a women’s Olympic beach volleyball match at London 2012 (ITA vs JPN). At ~0:02–0:03, the Japanese player on the right side executes a powerful spike toward Italy’s left court. The Italian blockers attempt to defend but fail; the ball lands cleanly inside the Italian court near the baseline without being touched again.

Verdict: Yes, the spike was successful — the ball landed inbounds on the opponent’s side, resulting in a point for Japan.

---

### qa_id=30532  ·  volleyball / Counterfactual
- **Q:** Would the first spike of the left team be successful if the other team did not save the ball?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** No. The first spike of the left team was unsuccessful because the opposing team blocked it. This indicates that the ball was not saved by the defense. Therefore, the correct answer is that the first spike of the left team was unsuccessful.

  **AFTER:**  The clip shows a women’s indoor volleyball match between Korea (KOR) and Thailand (THA), with KOR leading 2–1 in sets and 2–1 in points in the current set. The camera angle is wide, showing both sides of the court during active play.

At approximately 0:08–0:09, the Thai team (on the right side of the net) executes a defensive dig — one player dives and keeps the ball in play. Immediately after this dig, the Korean player on the left side (wearing white jersey #7) jumps at the net and spikes the ball sharply downward toward the Thai side.

If we ignore the dive shown by the Thai defender and f

---

### qa_id=30531  ·  volleyball / Counterfactual
- **Q:** Would the left team success to do their first spike if the other team did not save the ball?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** No. The left team failed to score because the right team saved the ball during the spike.

  **AFTER:**  Yes.

At roughly 0:08–0:09, the Korean player on the left side jumps at the net and executes a powerful overhead spike toward the Thai court. The Thai blockers attempt to contest but fail to stop the ball; it lands cleanly inside the opponent’s boundary lines without being touched again. That is the first offensive action shown in this sequence.

If the Thai players had not attempted to dig the ball, the spike would have resulted in a direct point for Korea. The only reason the rally continues is that the Thai defenders successfully kept the ball alive. Therefore, the spike itself succeeds as 

---

### qa_id=29442  ·  volleyball / Counterfactual
- **Q:** Would the spike of the left team be successful if the other team did not save the ball?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** No. The spike lands on the net, resulting in a loss for the left team. This indicates that the right team successfully saved the spike. Therefore, without saving the spike, the left team would have lost.

  **AFTER:**  Yes.

What I see: At roughly 0:00–0:01, the Brazilian player on the right side executes a powerful spike from the front row toward the back-left corner of the U.S. court. The trajectory is steep and aimed well inside the boundary lines. A U.S. defender dives and makes contact with the ball just before it hits the floor.

Why it matters: The spike’s angle and speed indicate it would have landed cleanly in bounds had there not been a defensive dig. Without that save, the ball would have struck the court surface within the limit, resulting in a point for Brazil.

---

### qa_id=31020  ·  volleyball / Counterfactual
- **Q:** Would the left team success to do spike if the other team did not save the ball?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** No. The left team could not succeed on the spike because the right team saved the ball.

  **AFTER:**  Yes.

At roughly 0:04–0:06, the French player on the left side (jersey #7) sets up for a spike near the net. The Australian defender dives and makes contact with the ball just before it hits the court, keeping it alive. Without that defensive touch, the ball’s trajectory and placement would have landed cleanly inbounds on Australia’s side—resulting in a successful spike. So, absent the save, the attack succeeds.

---

### qa_id=31021  ·  volleyball / Counterfactual
- **Q:** Would the spike of the left team be successful if the other team did not save the ball?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** No. The spike of the left team would likely fail because the right team’s defender blocked it.

  **AFTER:**  Yes.

At roughly 0:08–0:09, the French player on the left side executes a powerful spike aimed at the back-right corner of Australia’s court. The Australian defenders are already positioned; one dives and makes contact with the ball just before it hits the floor, keeping it alive. Without that dig, the trajectory and speed indicate the ball would have landed cleanly inbounds on Australia’s side—no block or net interference is visible. So, absent the save, the spike results in a point for France.

---

### qa_id=29695  ·  volleyball / Descriptive
- **Q:** Does any team score in the video?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** No, neither team scored. The scoreboard shows a tie at 7-7, and there is no visual indication of points being awarded during the clip.

  **AFTER:**  The clip shows a men’s volleyball match between Iran (IRI) and South Korea (KOR). At the start, IRI leads 2–7 in sets.

At approximately 0:00–0:01, the Korean player on the right side of the court executes a powerful spike toward the Iranian left side. The Iranian players attempt to block but fail; one defender dives but cannot stop the ball. The ball lands cleanly inside the boundary lines on the Iranian side of the court, untouched by any other player.

This is a direct kill — the spike results in a point for South Korea because the ball touched the floor within bounds without being returned

---

### qa_id=30500  ·  volleyball / Counterfactual
- **Q:** Would the right team success to do their first spike if the other team did not save the ball?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** The right team would fail to do their first spike because the left team saved the ball at the net. This is evident from the scoreboard showing China leading 2-5, indicating they were already winning the rally. The left team’s successful save prevented any further attempts on the court, which directly contradicts the premise that they could have successfully hit the ball without saving it. Therefore, the correct answer is False.

  **AFTER:**  Yes.

At roughly 0:01–0:02, the Chinese player on the right side executes a powerful spike toward the Serbian court. The Serbian defenders are already positioned; one dives and makes contact with the ball just before it hits the floor, preventing a direct point. Without that defensive save, the trajectory and speed of the spike would have landed cleanly in bounds on the opponent’s side, resulting in a successful attack for China. So, absent the save, the right team (China) would have scored.

---

### qa_id=29669  ·  volleyball / Descriptive
- **Q:** Does the left team do spike successfully?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** No, the left team does not spike successfully. The ball goes past their net during the spike attempt.

  **AFTER:**  Yes, the left team (Japan) executes a successful spike.

At approximately 0:06–0:07, the Japanese player on the front row jumps at the net and strikes the ball sharply downward toward Peru’s right side. The Peruvian defenders attempt to block but fail; one dives but cannot reach the ball. By 0:08, the ball lands cleanly inside Peru’s court near the backline, untouched by any defender. The trajectory and placement indicate a direct kill. No scoreboard or ref signal is needed—the visual outcome confirms the spike was successful.

---

### qa_id=30069  ·  volleyball / Counterfactual
- **Q:** Would the second spike of the left team be successful if the other team did not save the ball?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** The first spike of the left team was unsuccessful because the right team blocked it. Therefore, the second spike of the left team could not have been successful under those conditions.

  **AFTER:**  Yes.

At roughly 0:01–0:02, the Russian player on the left side executes a powerful spike from the front row. The Japanese blockers attempt to contain it but fail to fully stop the ball; one defender dives and makes contact, keeping the rally alive.

If the dive had not occurred, the trajectory and speed of the spike indicate it would have landed cleanly in the opponent’s court without any touch. Therefore, the spike would have been successful.

---

### qa_id=30770  ·  volleyball / Descriptive
- **Q:** Does the left team do save successfully?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** No, the left team does not save successfully. The net catches the ball, indicating that the right team saved the shot.

  **AFTER:**  The clip shows a men’s Olympic volleyball match at London 2012 between Bulgaria (BUL) and Russia (RUS). The camera cuts from a close-up of a Russian player to a wide view of the court during active play.

At approximately 0:04–0:06, a Bulgarian player (in red jersey #17) executes a powerful spike from the right side of the court. The ball travels sharply downward toward the Russian backcourt. Two Russian defenders attempt to block or dig but fail — one dives but cannot reach it; another is out of position. The ball lands cleanly inside the boundary lines on the Russian side, untouched by any d

---

### qa_id=30111  ·  volleyball / Descriptive
- **Q:** Does the right team do defend successfully?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** No, the right team does not defend successfully. The ball is hit back into the net by the left team.

  **AFTER:**  Yes.

At roughly 0:02–0:03, the Korean player on the right side sets up a block at the net while two Indian defenders jump to cover. The ball lands cleanly inside the court near the sideline, untouched by any defender. That is a successful block that results in a point for Korea.

---

### qa_id=29863  ·  volleyball / Counterfactual
- **Q:** Would the left team success to do spike if the other team did not save the ball?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** The left team could potentially attempt a spike, but they would likely fail due to the defense’s defensive strategy. The right team successfully saved the ball during the spike attempt, preventing the attack. Therefore, the left team did not succeed in doing the spike.

  **AFTER:**  Yes.

At roughly 0:06–0:08, the Croatian player on the right side sets up for a spike. The Canadian defender dives and makes contact with the ball just before it hits the net, keeping it in play. Without that dive/save, the trajectory and placement of the spike would have landed cleanly inside the court boundary—no block or dig would have altered its path. So, had the opposing team not saved the ball, the spike would have resulted in a point for the left team (Canada).

---

### qa_id=29507  ·  volleyball / Descriptive
- **Q:** Is the serve of the left team successful?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** No. The serve of the left team was unsuccessful because the ball bounced off the net before reaching the right side of the court.

  **AFTER:**  The clip shows a women’s indoor volleyball match at the VNL Finals between USA and CHN. At roughly 0:04–0:06, the player on the left side (USA) executes a jump serve that travels over the net toward the right court. The Chinese defenders attempt to block but fail; one player dives and cannot reach the ball. The ball lands cleanly inside the boundary lines on the opponent’s side without being touched again.

Evidence: The ball clearly bounces once on the floor within bounds after being served by the USA player. No touch occurs after the serve.

Verdict: Yes, the serve was successful. It resulte

---

### qa_id=31133  ·  volleyball / Counterfactual
- **Q:** Would the second spike of the right team be successful if the other team did not save the ball?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** The first spike by the German team was unsuccessful because the ball bounced back into their own court. The second spike by the USA team, which landed high in the air, also failed to clear the net due to the bounce. Therefore, if the other team had saved the ball at that moment, they could have avoided the double-bounce and potentially scored. However, this conclusion relies on the assumption that the ball remained in play during the subsequent shots, which contradicts the visual evidence showing both balls bouncing back. Thus, the correct answer is no; the second spike would not have been suc

  **AFTER:**  The clip shows a men’s Olympic beach volleyball match at London 2012. The USA team is on the left side of the court; Germany (GER) is on the right.

At approximately 0:01–0:02, the German player in red (#7) executes a powerful spike from the right front position. The ball travels sharply downward toward the U.S. side near the sideline. At that instant, two U.S. defenders are already positioned under the trajectory of the spike — one dives forward to attempt a dig, while another watches closely.

If the diving defender had not made contact with the ball — i.e., if he had not saved it — the ball

---

### qa_id=31130  ·  volleyball / Counterfactual
- **Q:** Would the right team success to do their first spike if the other team did not save the ball?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** The right team could potentially succeed in scoring their first spike if the other team did not save the ball during the rally. However, this depends on the exact timing of the save attempt and the subsequent sequence of events in the match. Without more context about the immediate aftermath of that save, we cannot definitively predict the outcome based solely on the provided information. The question requires analyzing the full sequence of plays and outcomes beyond just the initial spike attempt.

  **AFTER:**  Yes.

At roughly 0:01–0:02 in the wide view, the German player on the right side executes a powerful spike aimed at the front-left corner of the U.S. court. The trajectory is steep and well-placed toward the empty space near the net. At that exact moment, two U.S. defenders are positioned but out of position; one dives forward and attempts a dig, redirecting the ball upward.

If we remove the defensive save — i.e., assume the U.S. players didn’t make contact with the ball — the spike would have landed cleanly inside the boundary lines on the U.S. side. There is no indication the ball was block

---

### qa_id=31132  ·  volleyball / Counterfactual
- **Q:** Would the right team success to do their second spike if the other team did not save the ball?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** No. The first player on the court successfully sets up his second spike, but he does not receive the ball. The opposing team’s player intercepts the ball mid-air and returns it to the first player, who then attempts another spike attempt that fails due to lack of possession. Therefore, even without the opponent saving the ball, the first player cannot complete the set.

  **AFTER:**  The clip shows a men’s Olympic beach volleyball match at London 2012 between USA (in red) and Germany (in blue). The scoreboard reads USA 4–3 GER.

At approximately 0:01, the German player on the right side of the court executes a powerful spike toward the American left-side zone. At that exact moment, two U.S. defenders attempt to block but fail; one dives and makes contact with the ball just as it hits the net — this is the “save” referenced in the question.

If we remove that defensive touch and assume no save occurred — i.e., the ball continues unimpeded after the spike — the trajectory cl

---

### qa_id=30886  ·  volleyball / Counterfactual
- **Q:** Would the left team success to do their first spike if the other team did not save the ball?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** No, because the right team had already saved the ball before the left team’s spike could be made.

  **AFTER:**  Yes.

At roughly 0:04–0:06, the player on the left side (in red) executes a jump set near the net and spikes down toward the right court’s back corner. The trajectory is steep and aimed well inside the boundary lines; the ball clearly strikes the floor without being touched by any defender.

The only reason the rally continues is that a defender dives and makes contact with the ball just before it hits the court. Without that defensive touch, the spike lands cleanly in bounds—resulting in a direct point for the attacking team.

Verdict: Yes, the spike would have succeeded.

---

### qa_id=29825  ·  volleyball / Descriptive
- **Q:** Does any team gain a point?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** No. The scoreboard shows CAN 1 - 2 CRO at this moment, indicating that neither team has gained a point yet.

  **AFTER:**  Yes.

At roughly 0:02–0:03, Canada (CAN) sets up an attack; their middle blocker (#9) jumps at the net and spikes the ball down into Croatia’s court near the sideline. The Croatian defenders attempt to dig but cannot keep the ball in play—the ball lands cleanly on the court inside the boundary lines. That is a direct kill for Canada.

Evidence:
- Canadian player #9 executes a powerful spike that contacts the ball directly on the opponent’s side.
- The ball touches the playing surface within bounds without being touched again by any defender.
- No scoreboard update is needed to confirm the poin

---

### qa_id=29426  ·  volleyball / Counterfactual
- **Q:** Would the spike of the right team be successful if the other team did not save the ball?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** No, the spike was unsuccessful because the left team saved the ball. The left team had a high center and successfully blocked the shot.

  **AFTER:**  Yes.

At roughly 0:01–0:02, the Wisconsin player on the right side executes a powerful spike aimed at the backcourt near the sideline. The trajectory is steep and well inside the boundary lines; there’s no block deflection or net interference visible. The only reason the rally continues is that a Stanford defender dives and makes contact with the ball just before it hits the floor. Without that dive, the ball would have struck the court cleanly within bounds, resulting in a kill. So, absent the save, the spike would have been successful.

---

### qa_id=93338  ·  fg / Descriptive
- **Q:** What is the video about?
- **Gold:** `Balance Beam`
- **sim:** 0.583 → **0.833**  (Δ +0.250)

  **BEFORE:** The video features a female athlete performing on an apparatus bar at a competition venue. She starts by raising her arms in celebration, followed by executing a split jump onto the bar. The athlete then performs a series of jumps, including a round-off, a backhand jump, and a front handstand. Her movements are fluid and precise, showcasing her athletic skill and control. The setting includes a large screen displaying the event details and branding.

  **AFTER:**  This video captures a routine from an artistic gymnastics team competition. The focus is on a female athlete performing a synchronized sequence on the floor exercise apparatus.

At the start of the clip, we see the athlete standing behind the apparatus, facing away from the camera. She is wearing a black leotard with pink accents and the number "14" on her back. Behind her, a large screen displays the event branding and other competitors' names.

The scene then cuts to a wider view of the competition arena. The same athlete, now fully extended into a seated position on the floor, is executing 

---

### qa_id=28351  ·  aerobic_gymnastics / Descriptive
- **Q:** What is the video about?
- **Gold:** `aerobic gymnastics`
- **sim:** 0.778 → **1.000**  (Δ +0.222)

  **BEFORE:** The video shows two women performing aerobics on a gym floor during the 7th FIG Aer Gymnastics World Age Group Competitions incheon in June 2016. They execute various aerial maneuvers including jumps, spins, and flips while maintaining their form throughout the sequence. The setting includes banners indicating the event's location and date, emphasizing its professional nature. This appears to be part of a competitive gymnastics display showcasing individual skill and athleticism in this specific sport.

  **AFTER:**  The video shows two female rhythmic gymnasts performing on a competition floor during the "7th FIG Aerobic Gymnastics World Age Group Competitions" held in Incheon, Korea, from June 13–19, 2016. The backdrop includes sponsor logos such as Gachron University and VTB.

At the start of the clip (0:00), both athletes execute synchronized leaps with arms raised overhead — a classic element in aerobic gymnastics routines that tests flexibility, coordination, and power. Their leotards match in design but differ slightly in pattern; one wears black-and-white floral prints while the other has silver se

---

### qa_id=92844  ·  fg / Temporal
- **Q:** What do the players perform after performing transition flight from high bar to low bar?
- **Gold:** `transition flight from low bar to high bar`
- **sim:** 0.619 → **0.833**  (Δ +0.214)

  **BEFORE:** The players execute a swing into the high bar and then transition to the low bar, preparing for their next move on the apparatus.

  **AFTER:**  The clip shows two female rhythmic gymnasts performing on the apparatus (a ribbon) at the P&G Championships.

At approximately 0:00–0:01, one athlete executes a handstand-like position while the other performs a leap (transition flight) from a high bar to a low bar. Immediately following this motion, both athletes land on the mat and begin their next sequence.

After the transition flight and landing, they immediately move into a synchronized acrobatic element: a **round-off**. This is evident from their bodies extending forward and backward in a cartwheel-like motion, which serves as the prep

---

### qa_id=33782  ·  basketball / Descriptive
- **Q:** Which team scores in the video?
- **Gold:** `right team`
- **sim:** 0.600 → **0.800**  (Δ +0.200)

  **BEFORE:** The USA team scores. The scoreboard shows "USA 54 CAN 41" at the start of the clip, indicating that the United States has scored more points than the Canadian opponent. As the video progresses, the USA player drives to the basket and makes a successful layup, which is likely the scoring event being referenced. The Canadian player remains on defense throughout this sequence. Therefore, based on the visual data provided, the USA team is the one who scores.

  **AFTER:**  The clip shows a men’s indoor basketball match between USA (in white jerseys) and CAN (in dark blue/black jerseys). The scoreboard reads USA 54, CAN 41, with the timer at 03:03.

At approximately 0:04–0:05, a Canadian player (dark jersey #7) executes a jump serve from behind the end line. The ball travels high and fast toward the American side. Two U.S. defenders attempt to block but fail; one dives but cannot reach it. The ball lands cleanly on the U.S. court near the baseline, untouched by any other player.

This is a direct kill — the serve lands inbounds without being returned. In basketba

---

### qa_id=30749  ·  volleyball / Causal
- **Q:** What is the cause of the right team doing defend?
- **Gold:** `volleyball dink`
- **sim:** 0.600 → **0.800**  (Δ +0.200)

  **BEFORE:** The right team (Turkish team) is defending because they are positioned on the other side of the net and the serve has just landed there. The Turkish players are ready to receive the ball from the opposing team (Bahraini team) who has already served the ball into their own court. This defensive stance is necessary to prevent being hit by the incoming serve.

  **AFTER:**  The clip shows a women’s Olympic volleyball match at London 2012 (BRA vs TUR). At approximately 0:04–0:06, the left-side team (in black jerseys) executes a powerful spike toward the right side of the court. In response, the right-side team (yellow jerseys) immediately transitions into defensive positioning: players spread out, knees bent, hands ready to dig.

Cause: The right team defends because the left team has just executed a spike that is on course to land in their court. In volleyball, defense is reactive to an attack — specifically, when the opposing team threatens to score via a hit, t

---

### qa_id=27961  ·  aerobic_gymnastics / Temporal
- **Q:** What do the players perform after performing turn?
- **Gold:** `aerobic gymnastics illusion`
- **sim:** 0.630 → **0.815**  (Δ +0.185)

  **BEFORE:** The players perform a jump. They execute a high leap into the air, using their bodies to propel themselves upwards. This is likely part of a gymnastic routine, possibly involving aerial maneuvers like a vault or a flip. The jump is a distinct action that follows the initial movement.

  **AFTER:**  After executing the turn, both athletes immediately transition into a synchronized leap. Specifically, they perform a straddle jump (also known as a split leap) where their legs extend fully apart horizontally while airborne. This occurs at approximately the 0:01 mark in the clip, following the rotation of their bodies. The movement is clean and deliberate, showcasing flexibility and coordination typical of aerobic gymnastics routines.

---

## Regressions  (36)
Sorted by Δ gold-similarity (worst regression first).

### qa_id=28615  ·  aerobic_gymnastics / Descriptive
- **Q:** How many players are in the video?
- **Gold:** `2`
- **sim:** 1.000 → **0.000**  (Δ -1.000)

  **BEFORE:** 2

  **AFTER:**  There are two athletes performing in the video. They are both wearing matching yellow and black leotards and executing synchronized movements on the competition floor during the 5th Aerobic Gymnastics Asian Championships. No other performers enter the frame throughout the clip.

---

### qa_id=27897  ·  aerobic_gymnastics / Descriptive
- **Q:** How many types of actions do the players perform?
- **Gold:** `2`
- **sim:** 1.000 → **0.000**  (Δ -1.000)

  **BEFORE:** There are four distinct types of actions performed by the players in the video clip. These include:

1.  **Running:** The group moves forward across the court.
2.  **Dancing:** They execute synchronized movements that resemble a dance routine.
3.  **Jumping:** One player jumps into the air during the dance sequence.
4.  **Landing:** All participants return to their starting positions on the floor.

Therefore, there are four different types of actions observed in the clip.

  **AFTER:**  The clip shows three female rhythmic/aerobic gymnasts performing on a competition floor during the 5th World Age Group Competition in Aerobic Gymnastics.

At 0:00–0:01, they begin in a staggered line facing forward, arms raised, preparing for movement. At 0:01, they transition into synchronized dynamic motion — specifically, a leap or jump where one leg extends forward and the other backward mid-air, while their upper bodies lean forward and back in coordination. This is a classic “split leap” or “straddle jump,” common in aerobic gymnastics routines.

There is no second distinct type of actio

---

### qa_id=28899  ·  aerobic_gymnastics / Descriptive
- **Q:** Do the players perform turn?
- **Gold:** `no`
- **sim:** 1.000 → **0.500**  (Δ -0.500)

  **BEFORE:** No. The players execute a synchronized dance routine on a large gymnasium mat, moving their arms and bodies in coordinated patterns without any turns being performed.

  **AFTER:**  Yes. At roughly 0:00–0:01, all six athletes execute a full-body rotation while maintaining a straight line on the floor; their bodies rotate about the vertical axis, completing one or more turns in unison. This is a standard "turn" in aerobic gymnastics choreography.

---

### qa_id=34535  ·  basketball / Counterfactual
- **Q:** Would the left team success to do 2-point shot if the other team did not block?
- **Gold:** `no`
- **sim:** 1.000 → **0.500**  (Δ -0.500)

  **BEFORE:** The left team could potentially succeed to do a 2-point shot if the right team blocked. However, this is speculative based on the visual evidence alone without knowing the specific details of the game situation at that moment. The video shows a basketball game in progress, but there isn't enough information to definitively determine if the shot went in or not. Therefore, I cannot confirm the outcome with certainty.

  **AFTER:**  Yes.

At roughly 0:06–0:08 in the clip, the Turkish player on the left side (jersey #9) jumps at the net and attempts a two-point shot toward the Russian goal. The Russian blockers are already positioned under the shooter’s path. Without that block, the trajectory and angle of the shot would have landed cleanly inside the goal. The only reason the shot didn’t hit the goal is the successful block deflection; absent that, the action would have resulted in a point for Turkey.

---

### qa_id=34526  ·  basketball / Descriptive
- **Q:** What is the video about?
- **Gold:** `basketball`
- **sim:** 1.000 → **0.500**  (Δ -0.500)

  **BEFORE:** A basketball game between two teams, TUR and RUS, taking place at London 2012. The scoreboard shows TUR leading 16 to 28. A player in a white jersey is dribbling the ball while being closely guarded by a player in a red jersey. Another player in a red jersey jumps up and shoots the ball into the hoop. The camera follows the shooter as he moves towards the basket.

  **AFTER:**  This is a women’s Olympic indoor volleyball match from London 2012, specifically the second set between Turkey (TUR) and Russia (RUS). The scoreboard shows TUR leading 16–28 in points.

At the start of the clip, the Russian player #9 (in red) is positioned near the backcourt, preparing to receive a serve. A Turkish player (#16) is mid-air executing a jump serve — likely a “bump” or “tip” over the net. The ball is descending toward the Russian side.

The Russian players are already moving into position: one (#6) is skating forward under the ball, while others are spread out to cover possible re

---

### qa_id=34504  ·  basketball / Counterfactual
- **Q:** Would the right team success to do their first 2-point shot if the other team did not block?
- **Gold:** `no`
- **sim:** 1.000 → **0.500**  (Δ -0.500)

  **BEFORE:** The right team could not have done their first 2-point shot without blocking. The right team scored 41 points at this time, while the left team had 30. Therefore, they were already ahead and cannot succeed unless blocked.

  **AFTER:**  Yes.

At roughly 0:01–0:02, the blue-clad player on the right side (jersey #18) takes a jump approach and spikes the ball down near the net. The white-clad defenders on the left are already moving into position; one blocker jumps at the net and makes contact with the ball just as the spiker attempts to clear it. Without that block, the trajectory and placement suggest the ball would have landed cleanly inside the court—resulting in a successful two-point shot.

---

### qa_id=34565  ·  basketball / Counterfactual
- **Q:** Would the left team success to do 2-point shot if the other team did not block?
- **Gold:** `no`
- **sim:** 1.000 → **0.500**  (Δ -0.500)

  **BEFORE:** No, the left team could not successfully do a 2-point shot because they were blocked by the opposing team. The opposing player had already made contact with their own ball, preventing any further movement that would allow a successful layup.

  **AFTER:**  Yes.

At roughly 0:00–0:01, the French player on the right side takes a jump approach and spikes the ball toward the ESP goalkeeper’s zone. The Spanish defender dives but makes contact; without that dive, the ball would have struck the court cleanly inside the boundary lines—resulting in a successful 2-point shot. So, absent the block/dive, the left team (France) would have scored.

---

### qa_id=33631  ·  basketball / Descriptive
- **Q:** What are the players doing in the video?
- **Gold:** `basketball`
- **sim:** 1.000 → **0.500**  (Δ -0.500)

  **BEFORE:** The players are playing basketball on a court during a game. They are moving around the court, passing the ball, and trying to score points. The scoreboard shows that the USA team has 69 points while the China team has 38 points. The game is in the third quarter.

  **AFTER:**  This is a women’s Olympic beach volleyball match at London 2012 between China (CHN) and USA (USA). The scoreboard shows CHN leading 8–7 in the third set.

At the start of the clip, the U.S. player #9 is positioned near the net preparing to receive a serve. A Chinese player (#4) has just served the ball over the net. The U.S. players are already moving into position: two front-row players are spreading out to cover the court, while back-row defenders are ready to dig.

The key action occurs as the U.S. libero (#10) dives forward to attempt a defensive dig — but fails; the ball hits the floor in

---

### qa_id=33406  ·  basketball / Counterfactual
- **Q:** Would the 2-point shot of the right team be successful if the other team did not block?
- **Gold:** `no`
- **sim:** 1.000 → **0.500**  (Δ -0.500)

  **BEFORE:** No, because there was a defensive rebound immediately following the attempted shot. The opposing team recovered possession and scored the next point.

  **AFTER:**  Yes.

At roughly 0:02–0:03 in the clip, the player on the right side (in red) takes a jump shot toward the left corner of the court. The opposing blockers (two players in white) meet her at the net and deflect the ball back into play. Without that block, the trajectory and placement suggest the ball would have landed cleanly inside the boundary lines near the sideline—well within the 9-meter zone. So, absent the block, the shot would have been successful.

---

### qa_id=33810  ·  basketball / Descriptive
- **Q:** Does the left team do 3-point shot successfully?
- **Gold:** `no`
- **sim:** 1.000 → **0.500**  (Δ -0.500)

  **BEFORE:** No. The right team’s player executes a 3-pointer toward the basket. The left team does not attempt a 3-point shot.

  **AFTER:**  Yes.

At roughly 0:04–0:06, the Kentucky player on the left side executes a jump shot toward the right side of the court. The ball clearly passes cleanly through the net and lands inside the boundary lines on the opponent’s side without being touched by any defender. That constitutes a successful 3-point shot.

---

### qa_id=34066  ·  basketball / Descriptive
- **Q:** What is the video about?
- **Gold:** `basketball`
- **sim:** 1.000 → **0.500**  (Δ -0.500)

  **BEFORE:** The video shows a basketball game played in a professional arena during the 2018 Tenure de la France tournament. The scoreboard indicates that China leads the USA by 4 points in the third quarter at 9:17. Players are actively engaged in gameplay, with one player dribbling and passing while another jumps for a shot. The setting includes spectators and branding on the court, suggesting this is a high-level competitive match.

  **AFTER:**  This is a women’s indoor volleyball match from the FIVB World Championship tournament held in Tenerife, Spain, on June 18–20, 2018. The scoreboard shows China leading France 41–48 in set 3, with the current game score at 9:17.

The clip captures one complete rally:
- A Chinese player (in red jersey #16) receives the serve near the backcourt.
- She sets up an attack; her teammate (#10) approaches from the right side.
- Player #10 executes a powerful spike over the net toward the left side of the court.
- Two Spanish defenders attempt to block but fail to stop the ball.
- The ball lands cleanly 

---

### qa_id=33860  ·  basketball / Descriptive
- **Q:** What are the players doing in the video?
- **Gold:** `basketball`
- **sim:** 1.000 → **0.500**  (Δ -0.500)

  **BEFORE:** The players are playing basketball on a court during a match. They are actively engaged in gameplay, likely passing, shooting, or defending. The scoreboard indicates a competitive game between two teams.

  **AFTER:**  The clip shows a women’s Olympic beach volleyball match at London 2012, USA vs. FRA.

At the start of the sequence (0:00–0:01), the French player #8 is on the right side of the court near the net, preparing to receive a serve. The U.S. server (#14) has just tossed the ball into play; she is mid-air executing a jump serve toward the French side.

By 0:02, the ball is descending sharply toward the French court. A U.S. defender (#19) dives forward but cannot reach it cleanly — the ball hits her out of bounds on the left sideline. Simultaneously, another U.S. player (#17) begins walking away from 

---

### qa_id=33865  ·  basketball / Descriptive
- **Q:** What is the video about?
- **Gold:** `basketball`
- **sim:** 1.000 → **0.500**  (Δ -0.500)

  **BEFORE:** The video captures a dynamic moment during a basketball game. A player in a white uniform executes a powerful jump shot against opponents wearing red jerseys. The scene includes spectators seated in the stands, advertisements on the court walls, and officials monitoring the play. The action unfolds under bright arena lighting, highlighting the intensity and athleticism of the players involved in this high-stakes sporting event.

  **AFTER:**  This is a women’s indoor volleyball match between USA (in red) and FRA (in white), played during the 2016 Olympics in Rio. The scoreboard shows USA leading 12–9 in sets, with the current set score at 5–5.

At the start of the clip, the French player #8 (white jersey) receives the serve near the backcourt. She passes it to her setter (#14), who moves into position under the net. The setter sets up an attack; her teammate (#17) approaches from the left side, preparing to hit the ball over the net.

The U.S. blockers attempt to defend but fail to stop the spike. The French attacker executes a pow

---

### qa_id=34219  ·  basketball / Counterfactual
- **Q:** Would the 2-point shot of the left team be successful if the other team did not block?
- **Gold:** `no`
- **sim:** 1.000 → **0.500**  (Δ -0.500)

  **BEFORE:** No. The right team blocked the shot at the end of the second half, preventing the left team’s attempt.

  **AFTER:**  Yes.

At roughly 0:01–0:02, the USA player on the left side takes a jump shot toward the right side of the court. The Spanish blockers meet him at the net; one hand deflects the ball sharply upward and away from the goal. Without that block, the trajectory of the shot would have carried it cleanly into the goal. So, absent the block, the shot would have been successful.

---

### qa_id=83493  ·  fg / Temporal
- **Q:** Do the players perform gainer salto stretched with legs together at end of the beam before the salto backward stretched with 1.5 twist?
- **Gold:** `no`
- **sim:** 1.000 → **0.500**  (Δ -0.500)

  **BEFORE:** No, this does not occur. The sequence shown involves a forward stretch on the beam followed immediately by a backward stretch, but neither is a "gainer salto stretched with legs together" performed before the backward stretch. Additionally, there is no 1.5 twist element present in either movement segment described.

  **AFTER:**  Yes.

At roughly 0:00–0:01, the female athlete executes a gainer salto where she extends both legs forward while airborne; this is the "legs together" position described. Immediately after landing that jump, she transitions into a backward stretch on the beam. The question asks if they performed the gainer salto with legs together *before* the backward stretch. Since the gainer salto is clearly shown immediately prior to the backward stretch, the sequence is correct.

---

### qa_id=85055  ·  fg / Temporal
- **Q:** Do the players perform split leap forward before the switch leap to ring position?
- **Gold:** `no`
- **sim:** 1.000 → **0.500**  (Δ -0.500)

  **BEFORE:** No, the players do not perform a split leap before switching into ring position. The sequence begins with one athlete performing a split jump forward, which is followed by another athlete taking a split leap forward as well, both leading up to their respective ring jumps. Therefore, they both perform splits before switching positions, but there is no prior split leap performed by the second athlete.

  **AFTER:**  Yes.

At roughly 0:00–0:01, the athlete executes a split leap forward while airborne. Immediately after landing that leap, she transitions into a "switch leap" (a handstand-like position where legs are split) at approximately 0:01–0:02. The sequence is clear: split leap → transition → switch leap. Therefore, the split leap occurs directly before the switch leap.

---

### qa_id=93988  ·  fg / Temporal
- **Q:** Do the players perform leap forward with 0.25 turn into straddle pike position or side split to land on one or both feet after the stalder backward to handstand?
- **Gold:** `no`
- **sim:** 1.000 → **0.500**  (Δ -0.500)

  **BEFORE:** The players do not perform a leap forward with 0.25 turn into straddle pike position or side split to land on one or both feet after the stalder backward to handstand. The clip shows a female gymnast performing on the uneven bars, executing various moves including a backhand swing and a jump over the bar. There is no indication of a leap forward or a specific landing technique like a straddle pike or side split being performed immediately after the initial stance.

  **AFTER:**  The athlete performs a back handspring (or similar acrobatic element) that lands on her right foot. Immediately following this, she executes a full rotation on the ground, transitioning into a straddle pike position (legs wide apart, torso leaning forward) while keeping her arms extended. This movement is a standard part of the choreography in rhythmic gymnastics or artistic group routines, often used to demonstrate flexibility and control. The key here is the immediate transition from the inverted handstand to the horizontal straddle pike without any pause. Therefore, the correct answer is: Y

---

### qa_id=82568  ·  fg / Descriptive
- **Q:** What is the video about?
- **Gold:** `Balance Beam`
- **sim:** 1.000 → **0.500**  (Δ -0.500)

  **BEFORE:** The video features a gymnast performing on the balance beam at an event. She starts by standing on her feet, then jumps onto the beam. As she begins her routine, she performs various moves, including splits, turns, and other gymnastic maneuvers. The setting appears to be a competition venue, indicated by the large American flag backdrop and professional lighting. Her movements are fluid and controlled, showcasing her skill and athleticism. Throughout the clip, there are no other people visible, focusing solely on the gymnast's performance. The overall impression is one of precision and discipl

  **AFTER:**  This is a rhythmic gymnastics routine performed by a female athlete on a competition floor. The sequence begins with the gymnast executing a split leap while holding her ribbon. She immediately transitions into a synchronized tumbling pass, performing a round-off followed by a back handspring and a layout element. Throughout this segment, she maintains high levels of flexibility and coordination, showcasing her strength and agility as she moves across the apparatus. The focus is on demonstrating control, precision, and artistic expression typical of rhythmic gymnastics routines.

---

### qa_id=33096  ·  football / Descriptive
- **Q:** Does the right team do cross successfully?
- **Gold:** `no`
- **sim:** 1.000 → **0.500**  (Δ -0.500)

  **BEFORE:** No, the right team does not successfully make a cross. The ball remains in possession of the left team throughout the entire sequence shown.

  **AFTER:**  The clip shows a women’s Olympic beach volleyball match at London 2012 between COL and PRK. At approximately 0:01–0:02, the player on the right side (in yellow jersey #18) executes a powerful overhead spike toward the left court. The defending players attempt to block but fail; the ball lands cleanly inside the opponent’s boundary lines near the sideline, untouched by any defender. This is a successful attacking play resulting in a point for the right team. Yes, the right team scores via a successful cross-spike.

---

### qa_id=31588  ·  football / Descriptive
- **Q:** Does any team gain a point?
- **Gold:** `no`
- **sim:** 1.000 → **0.500**  (Δ -0.500)

  **BEFORE:** No. The scoreboard shows RMA 3 - 1 FCB, indicating that FC Barcelona won the match. However, there is no visual evidence of any team gaining points during this clip; the focus remains on the ongoing gameplay.

  **AFTER:**  Yes, the team in white (FCB) gains a point.

At the start of the clip, the scoreboard reads RMA 3 - 1 FCB. The camera shows a wide view of the pitch during a UEFA match between Real Madrid (RMA) and FCB. A player in a white jersey receives the ball near the center circle and takes a run toward the goal. He passes the ball past the goalkeeper into the net. The players immediately react to the save; one player dives but fails to prevent the goal. Since the ball crossed the goal line cleanly after the pass, the attacking team scores a goal. In soccer rules, scoring a goal awards one point to the 

---

### qa_id=31828  ·  football / Counterfactual
- **Q:** Would the left team success to do shoot if the other team did not do saving?
- **Gold:** `no`
- **sim:** 1.000 → **0.500**  (Δ -0.500)

  **BEFORE:** The left team did not score because they were unable to save the shot. The goalkeeper blocked the ball, preventing any attempt on goal. This indicates that the defending team had superior control of the ball during this moment.

  **AFTER:**  Yes.

At roughly 0:04–0:06, the Leipzig player on the left side receives the ball near the penalty area and takes a low approach toward the goal. The Bayern goalkeeper dives and makes contact with the ball just as it crosses the line, preventing it from touching the goal. Without that save, the trajectory and placement of the shot indicate it would have landed cleanly inside the goal—resulting in a successful goal for Leipzig.

---

### qa_id=30611  ·  volleyball / Counterfactual
- **Q:** Would the right team success to do their first spike if the other team did not save the ball?
- **Gold:** `no`
- **sim:** 1.000 → **0.500**  (Δ -0.500)

  **BEFORE:** No. The right team would fail to execute their first spike because the opposing team had already saved the ball with a lob before it could reach the net. This means the opponent’s defensive strategy directly prevented the right team from scoring immediately on that attempt.

  **AFTER:**  Yes.

At roughly 0:04–0:05, the Canadian player on the right side executes a powerful spike aimed at the backcourt near the sideline. The Croatian defenders are positioned to dig it; one dives and makes contact with the ball just as it crosses the net. Without that defensive touch, the ball’s trajectory and speed indicate it would have landed cleanly in bounds on the opponent’s court. So, absent the save, the spike results in a point for Canada.

---

### qa_id=30153  ·  volleyball / Counterfactual
- **Q:** Would the spike of the right team be successful if the other team did not save the ball?
- **Gold:** `no`
- **sim:** 1.000 → **0.500**  (Δ -0.500)

  **BEFORE:** The spike would likely be successful. The ball is moving toward the net, which is designed to catch spikes. Even without a defensive player saving it immediately, the trajectory suggests the ball will hit the backboard or hoop area where the spike lands. This implies that the defense might not have been able to block the spike before it reached its target zone. Therefore, the answer is yes, the spike would be successful.

  **AFTER:**  Yes.

At roughly 0:04–0:06, the Russian player on the right side executes a powerful spike aimed at the Bulgarian backcourt near the endline. The Bulgarian defenders are already moving into position; one dives and makes contact with the ball just as it crosses the net. Without that dive-save, the trajectory and speed of the spike indicate it would have landed cleanly inbounds on the opponent’s court—resulting in a direct point for Russia. So, absent the defensive save, the spike would have been successful.

---

### qa_id=30229  ·  volleyball / Descriptive
- **Q:** Does the left team do defend successfully?
- **Gold:** `no`
- **sim:** 1.000 → **0.500**  (Δ -0.500)

  **BEFORE:** No. The left team does not defend successfully. They attempt to block the right-side attack but fail to stop the ball from passing back into their own court. Instead of defending effectively, they allow the ball to continue its trajectory toward the opponent’s side, resulting in a loss of possession. This indicates that defensive effort was ineffective against the attacking play.

  **AFTER:**  Yes.

At roughly 0:04–0:06, Brazil’s right-side defender dives to dig the Italian spike. The ball stays in play on Brazil’s side and is kept alive; the rally continues. So the defensive attempt succeeds.

---

### qa_id=82896  ·  fg / Descriptive
- **Q:** What is the video about?
- **Gold:** `Balance Beam`
- **sim:** 1.000 → **0.583**  (Δ -0.417)

  **BEFORE:** The video features a gymnast named Kara Eaker performing on a balance beam. She begins by standing at the end of the beam, preparing for her routine. As she starts her routine, she executes several moves including a round-off, a fork, and a jump. The video captures her movements in detail, showing her technique and form throughout each move. Her performance highlights her skill and dedication to the art of gymnastics.

  **AFTER:**  The video features a rhythmic gymnastics routine performed by Kara Eaker. She begins her sequence on the floor, executing a high-knee lift while wearing a yellow leotard with white lace trim. The camera cuts to a wider view showing her mid-air during a turn on the ribbon apparatus. She is dressed in matching gold and yellow leotards, which she wears throughout the routine.

At approximately 0:01–0:02, the focus shifts to her partner, who is also wearing a similar uniform. They perform synchronized movements, including arm extensions and leg lifts, demonstrating coordination and timing. The bac

---

### qa_id=33474  ·  basketball / Descriptive
- **Q:** What is the video about?
- **Gold:** `basketball`
- **sim:** 1.000 → **0.600**  (Δ -0.400)

  **BEFORE:** The video features a basketball game played on a court with players in yellow jerseys competing against those in dark blue uniforms. The scoreboard indicates that the USA team has scored 2 points while the AUS team has 0. The clock shows approximately 9 minutes and 37 seconds remaining in the first quarter. A player in a yellow jersey dribbles towards the basket, preparing to make a shot. The camera follows the action closely, highlighting the players' movements and positioning as they approach the hoop. The background includes spectators in the stands, adding to the atmosphere of the match. T

  **AFTER:**  This is a women’s Olympic indoor volleyball match between USA and Australia, played during the 2016 Rio Games. The scoreboard shows USA leading 2–0 in sets, with the current set score at 9–7 in favor of Australia.

At the start of the clip, the Australian player #8 (in yellow) is positioned near the backcourt on the right side, preparing to receive a serve. The U.S. server (#14, also in yellow) has just served the ball over the net. In response, the Australian players react: one moves forward to pass the ball while others shift position defensively.

The key moment occurs when the Australian p

---

### qa_id=33602  ·  basketball / Descriptive
- **Q:** What is the video about?
- **Gold:** `basketball`
- **sim:** 1.000 → **0.600**  (Δ -0.400)

  **BEFORE:** The video captures a dynamic moment during a basketball game at the World Cup 2019 in Guangzhou. It shows players actively engaged in the match, moving strategically on the court. The scoreboard indicates that the American National Basketball Association (ANG) is leading against the Philadelphia Warriors (PHI) with a score of 33-26 in the second quarter, with 4 minutes remaining. The footage highlights key gameplay actions such as dribbling, shooting, and defensive maneuvers, providing a glimpse into the competitive atmosphere of the tournament.

  **AFTER:**  This is a women’s Olympic indoor volleyball match from the World Cup Qualifier in Rio 2016, specifically the second set between Angola (ANG) and Philippines (PHI). The scoreboard shows ANG leading 33–26 in the first set, with PHI leading 4:21 in the current set.

The key moment occurs at approximately 0:01–0:02: A Filipino player (in white jersey #39, Tancea) executes a powerful spike from the right side of the court. She leaps high, contacts the ball mid-air, and directs it sharply downward toward the left corner of the Angolan defense. The ball lands cleanly inside the boundary lines on the 

---

### qa_id=34631  ·  basketball / Descriptive
- **Q:** What is the video about?
- **Gold:** `basketball`
- **sim:** 1.000 → **0.600**  (Δ -0.400)

  **BEFORE:** The video captures a basketball game during the Athens 2004 Olympics. It shows players on court, including one player who appears to be attempting a jump shot while being defended by another player. The scoreboard indicates that the opposing team has scored 60 points against the current team’s 54 points, indicating this is a significant lead. The setting includes spectators in the stands and officials at the sidelines, suggesting a professional match.

  **AFTER:**  This is a women’s Olympic beach volleyball match at Athens 2004, played in a standard indoor court setup with two teams competing. The scoreboard shows Italy leading 6–4 in the first set, with the current set score tied at 54–54.

At the start of the clip, the Italian player #18 (in blue) receives the serve near the backcourt. She sets up for an attack but instead executes a defensive dig — likely a “dig” or “roll” — keeping the ball alive. Her teammate (#7, also in blue) moves forward to cover the net area. Meanwhile, the U.S. players are positioned defensively on both sides of the court.

Th

---

### qa_id=34603  ·  basketball / Descriptive
- **Q:** What are the players doing in the video?
- **Gold:** `basketball`
- **sim:** 1.000 → **0.600**  (Δ -0.400)

  **BEFORE:** The players are playing basketball in a women's competition. They are performing various basketball moves such as shooting, dribbling, and passing. The camera focuses on one player who is taking a shot at the basket.

  **AFTER:**  The clip shows a women’s indoor volleyball match during the FIBA Women’s Asian Cup 2019, between Australia (AUS) and Philippines (PHI). The scoreboard reads AUS 70 – PHI 36 in the third set, with 6:10 remaining in the second half.

At approximately 0:00–0:01, we see a Filipino player (in yellow jersey #18) executing a powerful overhead spike from the right side of the court. She leaps near the net, arms fully extended, and strikes the ball sharply downward toward the Australian side. Two Australian defenders (#4 and #12) attempt to block but are out of position; one dives low but fails to dig 

---

### qa_id=34091  ·  basketball / Descriptive
- **Q:** What are the players doing in the video?
- **Gold:** `basketball`
- **sim:** 1.000 → **0.600**  (Δ -0.400)

  **BEFORE:** The players are playing basketball on a court during a professional match. The scoreboard indicates that the Philadelphia Warriors are leading the Indiana Pacers at home in the second quarter.

**Evidence:**
- The players are positioned on the basketball court.
- There are advertisements around the court.
- A scoreboard shows the time remaining in the game (7:28) and the current score (PHI 26 - IRI 22).
- Players are moving around the court, likely preparing for a play or continuing their own game.

**Verdict:**
Based on the visual evidence, the players are actively engaged in a basketball gam

  **AFTER:**  The clip shows a women’s indoor volleyball match during the World Cup Qualifier between Philadelphia (PHI) and Iran (IRI). The scoreboard indicates PHI leading 26–22 in the second set.

At the start of the sequence, a player in white (Iran) sets up near the net while teammates move into position. A player in blue (Philadelphia) executes a jump serve from behind the end line. The Iranian server hits the ball sharply downward toward the left side of the court.

Two Iranian blockers attempt to defend but fail to stop the serve; the ball lands cleanly inside the boundary lines on the Iranian side,

---

### qa_id=33703  ·  basketball / Descriptive
- **Q:** What are the players doing in the video?
- **Gold:** `basketball`
- **sim:** 1.000 → **0.600**  (Δ -0.400)

  **BEFORE:** The players in the video are playing basketball. They are actively running and dribbling the ball on the court. The scoreboard indicates that the USA team has 61 points and China has 36 points, which suggests this is a competitive match. The game is currently in the third quarter, as indicated by the clock. The players appear to be focused on their task, moving strategically across the court.

  **AFTER:**  The clip shows a women’s Olympic indoor volleyball match between China (CHN) and USA at London 2012. At the start, USA leads 6–3 in sets; here we’re on the third set, score tied 36–36.

At ~0.1s, the Chinese setter (in red #9) has just set the ball near the net for her teammate (#18, in white) who is approaching from the right side. The U.S. blockers are already moving into position.

By 0.7s, player #18 executes a jump serve: she tosses the ball high, contacts it mid-air, and strikes it downward toward the U.S. court. Simultaneously, other players react—some shift weight forward, others prepa

---

### qa_id=34319  ·  basketball / Descriptive
- **Q:** What is the video about?
- **Gold:** `basketball`
- **sim:** 1.000 → **0.600**  (Δ -0.400)

  **BEFORE:** This video captures a moment during a professional basketball game at the London 2012 Olympics. The scoreboard indicates that France leads Spain by two points, with the third quarter underway. A player in a red jersey is seen executing a jump shot against a defender in white. The camera follows the action closely, showing the player moving up the court while the opposing team prepares to intercept. The scene highlights the intensity and skill involved in high-stakes Olympic basketball competitions.

  **AFTER:**  This is a women’s Olympic indoor volleyball match from London 2012, specifically the third set between France (FRA) and Spain (ESP). The scoreboard shows FRA leading 37–34 in the first set, and ESP leading 34–37 in the second.

In the clip, we see a French player (in red jersey #8) executing a powerful spike from the right side of the court. The Spanish defenders (in white jerseys) attempt to block but fail — the ball lands cleanly inside their court boundaries near the sideline, untouched by any other player. This results in a direct point for France.

The key actions observed:
- French attac

---

### qa_id=86521  ·  fg / Descriptive
- **Q:** What is the video about?
- **Gold:** `Vault`
- **sim:** 1.000 → **0.600**  (Δ -0.400)

  **BEFORE:** The video captures a gymnast performing on an apparatus at a competition. She starts by standing on a platform, then executes a flip into a vaulted ring, which she lands on her feet. This is followed by another sequence where she performs a similar maneuver but this time without flipping, simply landing on the ground. The setting appears to be a competitive environment, indicated by banners and lighting typical of such events. The focus remains entirely on the gymnast’s movements and actions throughout these sequences.

  **AFTER:**  This is a rhythmic gymnastics group routine at the London 2012 Olympics. The clip shows a male athlete performing on the floor during a synchronized sequence. He executes a complex turn while simultaneously lifting his partner into the air. The female gymnast performs a split leap, demonstrating flexibility and grace. Both athletes wear matching red, white, and blue leotards with gold embellishments. The background displays "London 2012" banners and event branding, confirming the competitive context. The focus is on their coordination, synchronization, and individual skill execution within the

---

### qa_id=28769  ·  aerobic_gymnastics / Descriptive
- **Q:** Do the players perform straight jump?
- **Gold:** `yes`
- **sim:** 1.000 → **0.667**  (Δ -0.333)

  **BEFORE:** Yes. The players execute a high arc jump that arcs forward and slightly upward, consistent with a standard athletic jump. This motion does not follow a perfectly vertical trajectory; instead, it curves away from the ground plane, indicating a non-straight jump. Therefore, the answer is no.

  **AFTER:**  No.

At roughly 0–1 second into the clip, two athletes on the left side execute a synchronized leap where their legs extend fully apart mid-air while keeping their bodies relatively upright. This is a straddle jump. A straight jump requires the legs to remain together throughout the jump; here, they are clearly separated. Therefore, the players do not perform a straight jump.

---

### qa_id=34401  ·  basketball / Descriptive
- **Q:** Does any team score in the video?
- **Gold:** `yes`
- **sim:** 1.000 → **0.667**  (Δ -0.333)

  **BEFORE:** Yes. The Russian team scored at the end of the clip, while the Turkish team lost.

  **AFTER:**  No.

At the start of the clip, Russia leads Turkey 25–16. The play shown is a defensive setup: a Russian player sets up near the net while two Turkish defenders attempt to block. A Russian player jumps at the net but fails to make contact; the ball deflects off her hands and lands on the Turkish side of the court. Since the ball touches the floor on the opponent’s side without being returned, this is a turnover for Russia. No scoring event occurs in this sequence.

---

### qa_id=29118  ·  aerobic_gymnastics / Descriptive
- **Q:** What is the video about?
- **Gold:** `aerobic gymnastics`
- **sim:** 0.833 → **0.722**  (Δ -0.111)

  **BEFORE:** The video features two individuals performing gymnastics moves on a wooden floor indoors. The first person executes a series of acrobatic movements including flips, spins, and jumps, while simultaneously executing another set of maneuvers such as cartwheels and twists. Both participants wear athletic attire suitable for gymnastics training or competition. Their movements suggest they are practicing specialized skills like aerial gymnastics, balance beam, or floor exercises. The setting appears to be a gymnasium or training facility with equipment and banners in the background indicating an org

  **AFTER:**  This is a rhythmic gymnastics group routine performed on a large carpeted mat inside an arena. The two athletes in red leotards execute synchronized acrobatic elements.

At the start of the clip, one gymnast performs a handstand while her partner does a cartwheel. Immediately following this, they transition into a dynamic sequence where both perform simultaneous leaps. Specifically, the gymnasts execute a split leap (one leg forward, one back) followed by a quick turn and another split leap. Their movements are fluid, high-energy, and perfectly timed to each other, showcasing their coordinatio

---

