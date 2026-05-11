# Flip-rate report — distillation before vs. after
Threshold (best_window_sim ≥): **0.8**  ·  n = **100** held-out samples
## Buckets
| bucket | n | share |
|---|---:|---:|
| **wrong → correct (FIX)** | 18 | 18.0% |
| correct → wrong (REGRESS) | 2 | 2.0% |
| correct → correct         | 22 | 22.0% |
| wrong → wrong             | 58 | 58.0% |

## Aggregate similarity
| metric | before | after | Δ |
|---|---:|---:|---:|
| correct rate (gold-fuzzy ≥ 0.8) | 0.240 | 0.400 | +0.160 |
| avg gold best-window sim | 0.599 | 0.781 | +0.182 |
| avg teacher-text sim     | 0.190 | 0.323 | +0.133 |

## By sport
| sport | n | fix | regress | persist_ok | persist_bad | correct_after |
|---|---:|---:|---:|---:|---:|---:|
| Gym | 50 | 9 | 1 | 8 | 32 | 0.340 |
| volleyball | 50 | 9 | 1 | 14 | 26 | 0.460 |

## By question_type
| qtype | n | fix | regress | persist_ok | persist_bad | correct_after |
|---|---:|---:|---:|---:|---:|---:|
| Causal | 13 | 0 | 0 | 0 | 13 | 0.000 |
| Counterfactual | 12 | 8 | 1 | 3 | 0 | 0.917 |
| Descriptive | 38 | 9 | 1 | 19 | 9 | 0.737 |
| Temporal | 37 | 1 | 0 | 0 | 36 | 0.027 |

## Flip-to-correct rows  (18)
Sorted by Δ gold-similarity (largest gain first).

### qa_id=760  ·  Gym / Descriptive
- **Q:** How many players are performing gymnastics?
- **Gold:** `2`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** Two players are performing gymnastics.

  **AFTER:**  In the provided clip, there are two female athletes performing on the floor during an aerobics gymnastics routine. Both are actively engaged in synchronized movements that include cartwheels, handstands, and other acrobatic elements typical of this discipline. No additional performers enter the frame during this segment.

Answer: 2 players are performing gymnastics.

---

### qa_id=23291  ·  volleyball / Descriptive
- **Q:** How many times does the left team do defend?
- **Gold:** `1`
- **sim:** 0.000 → **1.000**  (Δ +1.000)

  **BEFORE:** The left team does not do any defense in this clip; they only receive the serve. The right team performs all defensive actions during the rally. Therefore, the left team does not defend.

  **AFTER:**  In this volleyball clip from the London 2012 Olympics, we observe the following sequence:

At approximately 0:01–0:02, the Japanese player (in red) sets up an attack near the net. The Italian players (in blue) begin their defensive positioning — one player dives forward to dig the ball mid-air while others shift into ready stances. This is one coordinated defensive reaction.

No other defensive actions occur within the visible timeframe of the clip. The play ends immediately after the dig attempt; the ball is returned to Japan’s side, and the clip cuts off.

✅ Final Answer: **1 time** — the le

---

### qa_id=23971  ·  volleyball / Counterfactual
- **Q:** Would the dink of the left team be successful if the other team did not save the ball?
- **Gold:** `no`
- **sim:** 0.500 → **1.000**  (Δ +0.500)

  **BEFORE:** The dink of the left team would likely fail. The right team successfully saved the ball.

  **AFTER:**  The clip shows a women’s Olympic beach volleyball match at London 2012. The USA (in red) is on the left side; Brazil (in yellow) is on the right. At roughly 0:02–0:03, the Brazilian player near the net executes a soft “dink” (tip) aimed into the U.S. court.

Evidence: The ball is clearly airborne just above the U.S. side, descending sharply toward the open floor area where players are already moving to dig it. No U.S. defender is positioned under that trajectory at the moment of contact.

Verdict: Yes — the dink would have been successful if the other team had not saved the ball. It was an und

---

### qa_id=1948  ·  Gym / Descriptive
- **Q:** What are the players doing in the video?
- **Gold:** `aerobic gymnastics`
- **sim:** 0.556 → **1.000**  (Δ +0.444)

  **BEFORE:** The players are performing aerobics. They are moving in a synchronized fashion, which suggests they might be practicing their routine. The banner in the background mentions "Aerobic Incheon Korea" and "World Age Group Competitions," further confirming that this is an aerobics event.

  **AFTER:**  The clip shows three female rhythmic/aerobic gymnasts performing on a competition mat at the 14th FIG Aerobic Gymnastics World Age Group Championships in Incheon, Korea (June 16–19, 2016).

At 0:00–0:01, they execute synchronized arm extensions and leg lifts while maintaining upright posture — this is part of their choreographed routine. By 0:01, all three transition into dynamic movements: two perform forward rolls or cartwheels, while one executes a high leg extension (possibly a split or lunge) with arms raised overhead. Their leotards match, indicating team synchronization.

This is clearl

---

### qa_id=21412  ·  Gym / Descriptive
- **Q:** What are the players doing in the video?
- **Gold:** `aerobic gymnastics`
- **sim:** 0.611 → **1.000**  (Δ +0.389)

  **BEFORE:** The players in the video are performing a routine at an aerobics competition. They are executing various movements, including spins, jumps, and other dynamic actions characteristic of this type of athletic activity. The setting appears to be an indoor arena with banners indicating the event is part of a larger organized competition.

Based on the visual cues, they are likely demonstrating their skills in a competitive environment, showcasing their agility, coordination, and athleticism. This kind of practice could include drills like warm-ups, freestyle routines, or even acrobatic sequences ai

  **AFTER:**  The video shows four female athletes performing on a large mat inside an indoor arena. The backdrop banner reads "Aerobic Gymnastics."

At the start of the clip (0.0s–0.2s), all four athletes are moving in unison. They execute a synchronized leap where they extend their legs into a split position while airborne, followed immediately by landing back on the floor. This movement is known as a **split jump**.

They are performing a **group aerobic gymnastics routine**, which combines elements of dance, acrobatics, and strength training performed to music. The specific skill shown here is the **spl

---

### qa_id=3844  ·  Gym / Temporal
- **Q:** What do the players perform before performing helicopter?
- **Gold:** `aerobic gymnastics support`
- **sim:** 0.462 → **0.808**  (Δ +0.346)

  **BEFORE:** The players perform a split jump before performing a helicopter.

  **AFTER:**  In this aerobic gymnastics routine, two athletes begin their sequence on the mat. At approximately 0:04–0:06, both execute a synchronized handstand position while extending one leg forward and the other backward, supporting themselves on their hands and toes. This is technically known as a "handstand" or "one-arm handstand," depending on how the support is maintained.

Immediately following this handstand, they transition into a rotational movement where they spin horizontally with legs extended — this is the “helicopter” element. The handstand serves as the preparatory phase that generates mo

---

### qa_id=22608  ·  volleyball / Counterfactual
- **Q:** Would the first spike of the left team be successful if the other team did not save the ball?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** The first spike of the left team (Per) would likely fail against the right team (JPN) due to their defensive positioning and lack of defense on the net. The ball would bounce off the backboard and fall into the corner net, resulting in a loss. This is evident from the visual of the ball bouncing away from the right team's net.

  **AFTER:**  Yes.

At roughly 0:00–0:01, the left-side player (in red) sets up near the net and executes a powerful spike toward the right side of the court. The trajectory is steep and aimed well inside the opponent’s court boundary. At that instant, two defenders on the right are already moving laterally but don’t reach the ball; one dives and makes contact just before it hits the floor.

If the diving defender had not made contact, the ball would have landed cleanly inbounds on the right side of the court—no touch, no block deflection, no net interference. Therefore, the spike would have been successful

---

### qa_id=23653  ·  volleyball / Counterfactual
- **Q:** Would the left team success to do spike if the other team did not save the ball?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** No. The left team could not successfully execute a spike because they were unable to block the right team’s ball at that moment. A successful spike requires both teams to complete their respective shots without interference; since the left team blocked the ball, their attempt was invalid regardless of timing.

  **AFTER:**  Yes.

At roughly 0:04–0:06, the Brazilian player on the right side executes a powerful spike aimed at the backcourt near the sideline. The USA defender dives and makes contact with the ball just before it hits the court, keeping it alive. Without that dive-save, the trajectory and speed of the spike indicate it would have landed cleanly inbounds on the opponent’s side—resulting in a direct point for Brazil. So, absent the save, the spike succeeds.

---

### qa_id=23309  ·  volleyball / Counterfactual
- **Q:** Would the first spike of the left team be successful if the other team did not save the ball?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** The first spike of the left team would likely fail. The opponent, identified as Japan based on the scoreboard showing "JPN" vs "ITA", has already set up their own defense and appears ready to return the ball. Since they are defending against the player who just spiked the ball, they will almost certainly block that shot. Therefore, the first spike of the left team would not succeed.

  **AFTER:**  Yes.

At roughly 0:01–0:02, the Japanese player on the right side executes a powerful spike toward the Italian court near the net. The trajectory is clearly downward and aimed into open space on the Italian side. At that moment, two Italian defenders are already moving to block; one dives and makes contact just before the ball hits the floor, preventing the spike from touching the ground.

If the other team had not saved the ball—i.e., if no defender had made contact—the spike would have landed cleanly inbounds on the opponent’s court, resulting in a point for Japan. There is no visual indicat

---

### qa_id=8803  ·  volleyball / Counterfactual
- **Q:** Would the spike of the left team be successful if the other team did not save the ball?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** The spike of the left team was unsuccessful because the ball remained on the court and did not cross the net. The opponent saved it, resulting in a loss for that side.

  **AFTER:**  Yes.

At roughly 0:01–0:02, the red-clad player on the left side executes a powerful spike aimed at the backcourt corner near the sideline. The trajectory is steep and directed into open space with no blockers positioned there. At that instant, two defenders on the right side are already moving laterally but don't reach the ball; one dives and makes contact with it just as it hits the floor.

The spike lands cleanly inbounds without being touched again. There is no indication of a net touch or block deflection—this is a direct kill. Therefore, if the opposing team had not saved the ball, the s

---

### qa_id=6225  ·  volleyball / Counterfactual
- **Q:** Would the left team succeed in doing spike if the other team did not save the ball?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** No. The ball is caught by the right side of the court immediately after the spike attempt fails. This indicates that the opposing player saved the shot and moved the ball out of reach. Therefore, the left team could not have successfully executed a spike without saving the ball first.

  **AFTER:**  Yes.

At roughly 0:01–0:02, the Brazilian player on the right side executes a powerful overhead spike toward the U.S. court. The trajectory is steep and aimed at the back-left corner of the U.S. side. At that exact moment, two U.S. defenders are already moving into position; one dives low near the sideline while another attempts a block but is too late. The spiked ball is clearly below the net and well inside the boundary lines.

The key factor is timing and positioning: the U.S. players were reacting to the attack before it could be blocked or dug. Their defensive coverage was insufficient to

---

### qa_id=7715  ·  volleyball / Counterfactual
- **Q:** Would the right team succeed in doing their first spike if the other team did not save the ball?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** The left team did not save the ball during the spike attempt, leading to a missed shot. Therefore, the right team could not have achieved their first spike without saving the ball first. The correct answer is No.

  **AFTER:**  Yes.

At roughly 0:01–0:02, the Peruvian player on the right side executes a powerful overhead spike aimed at the backcourt near the endline. The Japanese defender dives and makes contact with the ball just before it hits the floor, keeping it alive. Without that dig, the trajectory and speed of the spike show it would have landed cleanly inbounds on Japan’s court. So, absent the save, the spike would have resulted in a point for Peru.

---

### qa_id=23170  ·  volleyball / Counterfactual
- **Q:** Would the right team success to do spike if the other team did not save the ball?
- **Gold:** `yes`
- **sim:** 0.667 → **1.000**  (Δ +0.333)

  **BEFORE:** No. The ball was already on the ground when the Japanese player jumped to spike it. Even without saving the ball, he could not reach it. He had to jump first to get the ball into his hands, which was why he couldn't perform the spike.

  **AFTER:**  Yes.

At roughly 0:01–0:02, the Japanese player on the right side executes a powerful overhead spike toward the backcourt. The Peruvian defender dives and makes contact with the ball mid-air, keeping it alive. Without that defensive touch, the trajectory and speed of the spike would have carried it cleanly into the court for a point. So, absent the save, the spike would have succeeded.

---

### qa_id=1432  ·  Gym / Descriptive
- **Q:** What are the players doing in the video?
- **Gold:** `aerobic gymnastics`
- **sim:** 0.722 → **1.000**  (Δ +0.278)

  **BEFORE:** The players are performing gymnastic routines on a wooden court at an international competition in Portugal. They are wearing black and yellow uniforms. The background shows banners for various organizations and flags. The setting appears to be a competitive environment.

  **AFTER:**  The video shows a group of five female athletes performing a synchronized routine on a competition mat during the FIG World Cup Aerobic Gymnastics event in Cantanhede, Portugal.

At the start of the clip (0.0–0.4 s), they are standing in formation, executing coordinated arm movements while maintaining leg extension and posture typical of aerobic gymnastics choreography. By 0.5 s, all five have transitioned into a low crouch position with hands planted firmly on the floor — this is a controlled descent into a plank-like pose, often used as a preparatory stance or transition element in aerobic r

---

### qa_id=3069  ·  Gym / Descriptive
- **Q:** What is the video about?
- **Gold:** `aerobic gymnastics`
- **sim:** 0.722 → **1.000**  (Δ +0.278)

  **BEFORE:** The video shows a group of female athletes performing on a gymnastics stage during the European Gymnastics Championships. The athletes are wearing pink leotards and black shorts. They are executing various gymnastic moves, including tumbling, acrobatics, and aerial maneuvers. The background features banners and logos related to the event.

  **AFTER:**  This is a clip from the **7th Aerobic Gymnastics European Championships**, held in Bologna, Italy.

The video shows a group of five female athletes performing on a competition mat. They are dressed in matching pink leotards and black shorts. The sequence begins with the athletes spread out across the floor, executing synchronized movements that include leg extensions and arm swings. As they move, they transition into more complex acrobatic elements such as handstands, cartwheels, and turns. Their movements are fluid and coordinated, showcasing their flexibility, strength, and timing.

At one p

---

### qa_id=3718  ·  Gym / Descriptive
- **Q:** What is the video about?
- **Gold:** `aerobic gymnastics`
- **sim:** 0.722 → **1.000**  (Δ +0.278)

  **BEFORE:** The video captures two female gymnasts performing synchronized aerial maneuvers on the balance beam during a competition at the People’s Republic of China 2018 World Championships. The event is held under the hashtag #AEROBICS_2018, indicating this is part of the aerobics program. Both athletes execute complex spinning and twisting movements while maintaining perfect alignment with their bodies, showcasing their strength, flexibility, and coordination. The setting features branding from sponsors like JOGOS SANTACASA and VTB, along with official flags representing Hungary and Russia, highlighti

  **AFTER:**  This video captures a routine from the 2018 Aerobic Gymnastics World Cup held in Hangzhou, China. The event features two athletes competing in the "People's Republic of China" category, Wang Zhou (left) and Zhou Wei (right).

The clip begins with both athletes standing side-by-side on the mat. They immediately transition into a synchronized sequence that includes handstands and leg lifts. Specifically, at approximately 0:01–0:02, they execute a dynamic movement where one athlete performs a full handstand while her partner supports her from behind with a high kick. This demonstrates strength, f

---

### qa_id=4103  ·  Gym / Descriptive
- **Q:** What are the players doing in the video?
- **Gold:** `aerobic gymnastics`
- **sim:** 0.778 → **1.000**  (Δ +0.222)

  **BEFORE:** The players are performing a synchronized routine on a gym floor, likely part of a competitive event. They start by stretching out their legs and bodies in various positions, possibly preparing for a specific move or technique. The setting appears to be an indoor arena with a large banner in the background that suggests this might be related to aerobics or gymnastics. As the sequence progresses, the players transition into more dynamic movements, indicating they are executing a coordinated series of actions typical of such athletic performances. The environment is well-lit, highlighting the mo

  **AFTER:**  The video shows a group of five female athletes performing on a competition floor marked with boundary lines. In the background, there is a banner that reads "Aerobic Gymnastics."

At the start of the clip, all five athletes are standing upright, facing forward. They begin moving into a synchronized formation. Specifically, they execute a cartwheel-like motion where they place their hands on the mat and kick their legs over their bodies. This is known as a **cartwheel**.

Immediately following the completion of the cartwheels, the athletes transition smoothly into a low crouch position, placin

---

### qa_id=21872  ·  Gym / Descriptive
- **Q:** What are the players doing in the video?
- **Gold:** `aerobic gymnastics`
- **sim:** 0.778 → **1.000**  (Δ +0.222)

  **BEFORE:** The players are performing a synchronized routine on a mat at a competition. They execute various gymnastic moves including jumps, spins, and flips while maintaining their positions and coordination. The background indicates that this event is part of the "7th FIG AER GYMNASTICS WORLD AGE GROUP COMPETITIONS" held in Incheon, Korea, in June 2016.

  **AFTER:**  The video shows a group of five female athletes performing a synchronized routine at the "7th FIG Aerobic Gymnastics World Age Group Competitions" in Incheon, Korea.

At the start (0:00–0:01), they execute a coordinated leap where two athletes jump into the air while others perform cartwheels or handstands on the floor. By 0:02, all five transition into a dynamic sequence involving high-energy movements such as jumps, turns, and acrobatic elements like flips and splits. The choreography emphasizes flexibility, strength, and synchronization—key components of aerobic gymnastics.

This is clearly

---

## Regressions  (2)
Sorted by Δ gold-similarity (worst regression first).

### qa_id=22332  ·  volleyball / Counterfactual
- **Q:** Would the left team success to do spike if the other team did not save the ball?
- **Gold:** `no`
- **sim:** 1.000 → **0.500**  (Δ -0.500)

  **BEFORE:** No. The spike lands on the opponent’s head during the rebound sequence. This indicates that the left team attempted the spike but failed due to the opposing player’s defense. Therefore, the spike did not succeed; instead, the opponent blocked the shot, resulting in a loss.

  **AFTER:**  Yes.

At roughly 0:04–0:06, the Russian player on the right side sets up an attack. The Bulgarian blockers meet him at the net; one defender dives and makes contact that deflects the ball back into play. Without that dive-save, the trajectory shows the spiked ball would have landed cleanly inside the court near the front row, resulting in a successful kill.

---

### qa_id=21027  ·  Gym / Descriptive
- **Q:** What is the video about?
- **Gold:** `aerobic gymnastics`
- **sim:** 0.889 → **0.722**  (Δ -0.167)

  **BEFORE:** The video shows four female athletes performing a synchronized routine on a gymnastics mat. They are dressed in matching outfits with blue tops and white bottoms. The background features banners indicating that this event is part of the "International Open Competition CANTANHEDE | PORTUGAL" and is held under the "FIG WORLD GYMNASTICS" umbrella. A scoreboard is visible on the screen behind them, displaying names and scores for each athlete. The athletes execute various moves including arm movements, jumps, and possibly tumbling passes, all while maintaining their positions. The setting appears 

  **AFTER:**  This is a rhythmic group gymnastics routine at the FIG World Cup in Cantanhede, Portugal. The clip shows five female athletes performing synchronized choreography on a wooden floor marked with black lines. They are wearing matching blue leotards with white accents and sparkly gold trim.

At the start of the clip, all five athletes are standing in formation facing forward, arms raised slightly. Then they transition into dynamic movement: two athletes execute high leg extensions while one performs a leap; another athlete moves laterally across the mat; and finally, two others perform coordinated

---

