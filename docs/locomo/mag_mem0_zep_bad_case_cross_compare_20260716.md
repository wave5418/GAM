# LOCOMO MAG/Mem0/Zep Bad Case Cross Comparison

## Scope
- mag: `/home/lhw/MAG/x/memory-benchmarks/results/locomo/locomo_results_20260716_175424.json`
  - project: `mag_full10_conservative_bfs_top30_top10_4omini_clean_20260716`
- mem0: `/home/lhw/MAG/results/locomo/locomo_results_mem0.json`
  - project: `mem0_local`
- zep: `/home/lhw/MAG/results/locomo/locomo_results_20260625_zep.json`
  - project: `graphiti_fastembed_top10_timectx`
- Aligned questions: `1540`
- Cutoff: `top_10`

## Initial Reading

- MAG-only failures are concentrated in retrieval differences: low TopK overlap with both baselines is the dominant signal.
- In MAG-only failures, lower ground-truth token support and graph-heavy TopK indicate evidence admission/ranking is a stronger suspect than answer generation alone.
- Mem0-only and Zep-only failures are useful controls: they show where summary-style memory or temporal fact/entity context loses detail even when MAG succeeds.
- Some judge-labeled correct cases still look like abstentions; treat those rows as evaluator-noise candidates before drawing method conclusions.

## Accuracy

| System | Correct | Total | Accuracy | Mean Latency ms | P50 ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|
| mag | 1319 | 1540 | 85.65% | 3763.31 | 3468.40 | 5817.20 |
| mem0 | 1298 | 1540 | 84.29% | 70.34 | 64.90 | 107.80 |
| zep | 1310 | 1540 | 85.06% | 167.75 | 170.50 | 235.90 |

## Pattern Counts

| Bucket | Count |
|---|---:|
| `all_correct` | 1050 |
| `mem0_wrong_mag_zep_correct` | 126 |
| `zep_wrong_mag_mem0_correct` | 103 |
| `mag_wrong_mem0_zep_correct` | 97 |
| `mag_zep_wrong_mem0_correct` | 48 |
| `mem0_zep_wrong_mag_correct` | 40 |
| `all_wrong` | 39 |
| `mag_mem0_wrong_zep_correct` | 37 |

## Requested Buckets

### MAG wrong, both baselines correct

- Count: `97`
- Question IDs: `conv0_q22, conv0_q29, conv0_q41, conv0_q44, conv0_q51, conv0_q55, conv0_q64, conv0_q71, conv0_q76, conv0_q90, conv0_q111, conv1_q16, conv1_q23, conv1_q31, conv2_q31, conv2_q75, conv2_q82, conv2_q98, conv2_q122, conv2_q147, conv2_q148, conv3_q29, conv3_q32, conv3_q52, conv3_q65, conv3_q69, conv3_q129, conv3_q134, conv4_q28, conv4_q32, conv4_q39, conv4_q40, conv4_q66, conv4_q72, conv4_q103, conv4_q108, conv4_q160, conv4_q161, conv4_q171, conv4_q174, conv5_q0, conv5_q5, conv5_q7, conv5_q15, conv5_q47, conv5_q56, conv5_q57, conv5_q58, conv5_q119, conv6_q2, conv6_q30, conv6_q43, conv6_q56, conv6_q61, conv6_q64, conv6_q95, conv6_q113, conv6_q123, conv6_q126, conv6_q141, conv6_q144, conv7_q36, conv7_q44, conv7_q55, conv7_q60, conv7_q121, conv7_q134, conv7_q140, conv7_q171, conv7_q172, conv8_q5, conv8_q6, conv8_q11, conv8_q41, conv8_q52, conv8_q53, conv8_q57, conv8_q60, conv8_q64, conv8_q67, conv8_q68, conv8_q72, conv8_q74, conv8_q95, conv8_q108, conv8_q110, conv8_q127, conv9_q11, conv9_q56, conv9_q57, conv9_q59, conv9_q68, conv9_q115, conv9_q127, conv9_q137, conv9_q154, conv9_q156`
- By category: `{'open-domain': 9, 'temporal': 31, 'multi-hop': 19, 'single-hop': 38}`
- Difference signals: `{'mag_gt_support_lower_than_correct_baselines': 65, 'mag_graph_heavy_when_wrong': 60, 'mag_low_topk_overlap_with_mem0': 97, 'mag_low_topk_overlap_with_zep': 97, 'zep_temporal_fact_context_available': 97, 'mag_much_slower_than_mem0': 97, 'mag_much_slower_than_zep': 97, 'zep_correct_but_answer_abstained': 15, 'mag_answer_abstained': 23, 'mem0_correct_but_answer_abstained': 4}`

- `conv0_q22` `open-domain` signals=`mag_gt_support_lower_than_correct_baselines,mag_graph_heavy_when_wrong,mag_low_topk_overlap_with_mem0,mag_low_topk_overlap_with_zep,zep_temporal_fact_context_available,mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: Would Caroline likely have Dr. Seuss books on her bookshelf?
  - GT: Yes, since she collects classic children's books
  - MAG: No, there is no evidence to suggest that Caroline would likely have Dr. Seuss books on her bookshelf.
  - Mem0: Yes, Caroline would likely have Dr. Seuss books on her bookshelf.
  - Zep: Yes, Caroline would likely have Dr. Seuss books on her bookshelf.
  - support MAG/Mem0/Zep: `0.0/0.2857/0.2857`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv0_q29` `temporal` signals=`mag_gt_support_lower_than_correct_baselines,mag_low_topk_overlap_with_mem0,mag_low_topk_overlap_with_zep,zep_temporal_fact_context_available,mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: When did Melanie go to the pottery workshop?
  - GT: The Friday before 15 July 2023
  - MAG: October 20, 2023
  - Mem0: Melanie went to the pottery workshop on July 7, 2023.
  - Zep: Melanie went to the pottery workshop on July 7, 2023.
  - support MAG/Mem0/Zep: `0.4/0.6/0.4`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv0_q41` `temporal` signals=`mag_gt_support_lower_than_correct_baselines,mag_graph_heavy_when_wrong,mag_low_topk_overlap_with_mem0,mag_low_topk_overlap_with_zep,zep_temporal_fact_context_available,mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: When did Caroline join a new activist group?
  - GT: The Tuesday before 20 July 2023
  - MAG: Caroline joined a new LGBTQ activist group on 17 October 2023.
  - Mem0: Caroline joined a new activist group on July 11, 2023.
  - Zep: Caroline joined a new activist group on July 20, 2023.
  - support MAG/Mem0/Zep: `0.4/0.6/0.4`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv0_q44` `temporal` signals=`zep_correct_but_answer_abstained,mag_gt_support_lower_than_correct_baselines,mag_answer_abstained,mag_low_topk_overlap_with_mem0,mag_low_topk_overlap_with_zep,zep_temporal_fact_context_available,mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: When is Melanie's daughter's birthday?
  - GT: 13 August
  - MAG: The specific date of Melanie's daughter's birthday is not mentioned in the memories.
  - Mem0: August 13, 2023.
  - Zep: The specific date of Melanie's daughter's birthday is not mentioned in the memories.
  - support MAG/Mem0/Zep: `0.0/1.0/0.0`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv0_q51` `multi-hop` signals=`mag_gt_support_lower_than_correct_baselines,mag_graph_heavy_when_wrong,mag_low_topk_overlap_with_mem0,mag_low_topk_overlap_with_zep,zep_temporal_fact_context_available,mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: What has Melanie painted?
  - GT: Horse, sunset, sunrise
  - MAG: Melanie has painted nature-inspired paintings, an abstract painting, and a calming piece.
  - Mem0: Melanie has painted a sunset-inspired painting with vibrant purple colors, a sunset-inspired landscape painting, a painting depicting a sunset with a pink sky, an abstract painting with vibrant colors, and an abstract painting aimed at portraying tranquility with peaceful blue streaks.
  - Zep: Melanie has painted a lake sunrise, a horse painting, an abstract painting, and a painting of a sunflower.
  - support MAG/Mem0/Zep: `0.0/0.3333/1.0`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv0_q55` `multi-hop` signals=`mag_gt_support_lower_than_correct_baselines,mag_low_topk_overlap_with_mem0,mag_low_topk_overlap_with_zep,zep_temporal_fact_context_available,mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: What subject have Caroline and Melanie both painted?
  - GT: Sunsets
  - MAG: Caroline has painted about her identity as a trans woman, while Melanie has painted abstractly to express emotions.
  - Mem0: Both Caroline and Melanie have painted sunsets.
  - Zep: Caroline and Melanie have both painted sunsets.
  - support MAG/Mem0/Zep: `0.0/0.0/1.0`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv0_q64` `open-domain` signals=`mag_gt_support_lower_than_correct_baselines,mag_graph_heavy_when_wrong,mag_low_topk_overlap_with_mem0,mag_low_topk_overlap_with_zep,zep_temporal_fact_context_available,mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: Would Melanie likely enjoy the song "The Four Seasons" by Vivaldi?
  - GT: Yes; it's classical music
  - MAG: No specific evidence suggests whether Melanie would enjoy "The Four Seasons" by Vivaldi.
  - Mem0: Yes, Melanie would likely enjoy the song "The Four Seasons" by Vivaldi.
  - Zep: Yes, Melanie would likely enjoy the song "The Four Seasons" by Vivaldi, as she is a fan of classical music.
  - support MAG/Mem0/Zep: `0.0/0.6667/0.6667`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv0_q71` `multi-hop` signals=`zep_correct_but_answer_abstained,mag_answer_abstained,mag_low_topk_overlap_with_mem0,mag_low_topk_overlap_with_zep,zep_temporal_fact_context_available,mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: What book did Melanie read from Caroline's suggestion?
  - GT: "Becoming Nicole"
  - MAG: The memories do not specify a book that Melanie read from Caroline's suggestion.
  - Mem0: Melanie read 'Becoming Nicole' by Amy Ellis Nutt from Caroline's suggestion.
  - Zep: Melanie is reading a book recommended by Caroline, but the title of the book is not specified.
  - support MAG/Mem0/Zep: `1.0/1.0/0.0`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv0_q76` `multi-hop` signals=`mag_gt_support_lower_than_correct_baselines,mag_graph_heavy_when_wrong,mag_low_topk_overlap_with_mem0,mag_low_topk_overlap_with_zep,zep_temporal_fact_context_available,mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: When did Melanie go on a hike after the roadtrip?
  - GT: 19 October 2023
  - MAG: There is no record of Melanie going on a hike after the road trip.
  - Mem0: October 19, 2023
  - Zep: Melanie went on a hike during the week of October 13-19, 2023, after the road trip on October 20, 2023.
  - support MAG/Mem0/Zep: `0.5/1.0/0.5`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv0_q90` `single-hop` signals=`zep_correct_but_answer_abstained,mag_gt_support_lower_than_correct_baselines,mag_graph_heavy_when_wrong,mag_answer_abstained,mag_low_topk_overlap_with_mem0,mag_low_topk_overlap_with_zep,zep_temporal_fact_context_available,mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: How long have Mel and her husband been married?
  - GT: Mel and her husband have been married for 5 years.
  - MAG: No information available about how long Mel and her husband have been married.
  - Mem0: 5 years
  - Zep: The exact duration of Mel and her husband's marriage is not specified in the memories.
  - support MAG/Mem0/Zep: `0.5556/0.7778/0.8889`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv0_q111` `single-hop` signals=`mag_gt_support_lower_than_correct_baselines,mag_low_topk_overlap_with_mem0,mag_low_topk_overlap_with_zep,zep_temporal_fact_context_available,mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: What creative project do Mel and her kids do together besides pottery?
  - GT: painting
  - MAG: Besides pottery, Melanie and her kids enjoy hiking in the mountains and exploring forests, which may inspire creative projects related to nature.
  - Mem0: Melanie and her kids work on nature-inspired paintings together besides pottery.
  - Zep: Melanie and her kids enjoy painting together, especially nature-inspired paintings.
  - support MAG/Mem0/Zep: `0.0/1.0/1.0`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv1_q16` `temporal` signals=`mag_gt_support_lower_than_correct_baselines,mag_low_topk_overlap_with_mem0,mag_low_topk_overlap_with_zep,zep_temporal_fact_context_available,mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: When did Jon go to a fair to get more exposure for his dance studio?
  - GT: 24 April, 2023
  - MAG: Jon did not go to a fair for exposure for his dance studio; there is no record of such an event in the memories.
  - Mem0: Jon went to a fair on April 24, 2023, to get more exposure for his dance studio.
  - Zep: Jon went to a fair on April 24, 2023, to show off his dance studio.
  - support MAG/Mem0/Zep: `0.5/1.0/0.5`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv1_q23` `multi-hop` signals=`mag_gt_support_lower_than_correct_baselines,mag_low_topk_overlap_with_mem0,mag_low_topk_overlap_with_zep,zep_temporal_fact_context_available,mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: How did Gina promote her clothes store?
  - GT: worked with an artist to make unique fashion pieces, made limited-edition sweatshirts, got some new offers and promotions for online store, developed a video presentation showing how to style her pieces
  - MAG: Gina promoted her clothing store by launching an ad campaign, emailing wholesalers, planning to work with fashion bloggers and influencers, and doing more ads.
  - Mem0: Gina promoted her clothing store by building relationships and creating a strong brand image, planning to work with fashion bloggers and influencers, running ads to reach a wider audience, and sharing a photo of her website featuring a sale ad.
  - Zep: Gina promoted her clothes store by launching an ad campaign, introducing new offers and promotions, working hard on her online store, and expanding her store after receiving positive feedback from a wholesaler.
  - support MAG/Mem0/Zep: `0.3704/0.3704/0.7407`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv1_q31` `multi-hop` signals=`mag_graph_heavy_when_wrong,mag_low_topk_overlap_with_mem0,mag_low_topk_overlap_with_zep,zep_temporal_fact_context_available,mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: How long did it take for Jon to open his studio?
  - GT: six months
  - MAG: It took Jon approximately 2 months to open his dance studio, from February 1, 2023, to April 3, 2023.
  - Mem0: Approximately 4 months and 18 days.
  - Zep: It took Jon approximately 4 months and 22 days to open his dance studio, from January 29, 2023, to June 20, 2023.
  - support MAG/Mem0/Zep: `0.0/0.0/0.0`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv2_q31` `temporal` signals=`zep_correct_but_answer_abstained,mag_graph_heavy_when_wrong,mag_answer_abstained,mag_low_topk_overlap_with_mem0,mag_low_topk_overlap_with_zep,zep_temporal_fact_context_available,mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: When did John get his dog Max?
  - GT: In 2013
  - MAG: The specific date when John got his dog Max is not mentioned in the memories.
  - Mem0: John got his dog Max around 2013.
  - Zep: The memories do not specify when John got his dog Max.
  - support MAG/Mem0/Zep: `0.0/0.0/0.0`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv2_q75` `single-hop` signals=`zep_correct_but_answer_abstained,mag_gt_support_lower_than_correct_baselines,mag_graph_heavy_when_wrong,mag_answer_abstained,mag_low_topk_overlap_with_mem0,mag_low_topk_overlap_with_zep,zep_temporal_fact_context_available,mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: What did Jean go through before meeting Maria?
  - GT: divorce, job loss, homelessness
  - MAG: No information is available about Jean's experiences before meeting Maria.
  - Mem0: Jean went through a divorce, lost her job, and became homeless before meeting Maria.
  - Zep: Jean went through a lot before meeting Maria, but specific details about her experiences are not mentioned.
  - support MAG/Mem0/Zep: `0.25/0.5/0.0`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv2_q82` `single-hop` signals=`mag_gt_support_lower_than_correct_baselines,mag_low_topk_overlap_with_mem0,mag_low_topk_overlap_with_zep,zep_temporal_fact_context_available,mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: What did John receive a certificate for?
  - GT: completion of a university degree
  - MAG: John did not receive a certificate; there is no mention of it in the memories.
  - Mem0: John received a certificate for the completion of a university degree.
  - Zep: John received a certificate for completing a university degree.
  - support MAG/Mem0/Zep: `0.0/1.0/0.6667`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv2_q98` `single-hop` signals=`mag_gt_support_lower_than_correct_baselines,mag_graph_heavy_when_wrong,mag_answer_abstained,mag_low_topk_overlap_with_mem0,mag_low_topk_overlap_with_zep,zep_temporal_fact_context_available,mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: What did John host for the veterans in May 2023 as part of the project?
  - GT: a small party to share their stories
  - MAG: John worked on a project to support veterans in May 2023, focusing on showing appreciation for their contributions, but the specific details of what he hosted are not mentioned.
  - Mem0: John worked on a project to show appreciation for veterans in May 2023, which included getting a petition going, but the specific event he hosted is not detailed in the memories.
  - Zep: John hosted a small party in May 2023 where he invited veterans to share their stories.
  - support MAG/Mem0/Zep: `0.4/0.4/1.0`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv2_q122` `single-hop` signals=`mag_graph_heavy_when_wrong,mag_low_topk_overlap_with_mem0,mag_low_topk_overlap_with_zep,zep_temporal_fact_context_available,mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: What motivated Maria and John to discuss potential solutions for their community on 7 July, 2023?
  - GT: Flood in John's old area
  - MAG: Maria and John were motivated to discuss potential solutions for their community on July 7, 2023, due to the issues of unemployment and the need for improvements in education and infrastructure.
  - Mem0: Maria and John were motivated to discuss potential solutions for their community on July 7, 2023, by their shared goal of helping others through community efforts and fundraising activities, as well as their mutual inspiration from each other's commitment to community improvement.
  - Zep: Maria and John were motivated to discuss potential solutions for their community on 7 July, 2023, due to John's desire to address community issues, particularly related to flood damage, and his appreciation for Maria's support and ideas.
  - support MAG/Mem0/Zep: `0.5/0.25/0.5`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv2_q147` `single-hop` signals=`mag_graph_heavy_when_wrong,mag_answer_abstained,mag_low_topk_overlap_with_mem0,mag_low_topk_overlap_with_zep,zep_temporal_fact_context_available,mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: What is the name of Maria's second puppy?
  - GT: Shadow
  - MAG: The name of Maria's second puppy is not specified in the memories.
  - Mem0: Shadow
  - Zep: Shadow
  - support MAG/Mem0/Zep: `1.0/1.0/0.0`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`
- Omitted examples: `77`; see CSV/JSONL outputs.

### Mem0 wrong, MAG/Zep correct

- Count: `126`
- Question IDs: `conv0_q1, conv0_q9, conv0_q34, conv0_q40, conv0_q43, conv0_q56, conv0_q60, conv0_q103, conv0_q110, conv0_q135, conv0_q138, conv1_q8, conv1_q26, conv1_q68, conv2_q25, conv2_q27, conv2_q32, conv2_q73, conv2_q84, conv2_q99, conv2_q102, conv2_q113, conv2_q114, conv2_q142, conv2_q151, conv3_q3, conv3_q4, conv3_q12, conv3_q27, conv3_q41, conv3_q51, conv3_q54, conv3_q56, conv3_q60, conv3_q61, conv3_q68, conv3_q70, conv3_q71, conv3_q73, conv3_q87, conv3_q90, conv3_q105, conv3_q117, conv3_q118, conv3_q165, conv3_q169, conv3_q180, conv3_q183, conv3_q189, conv4_q1, conv4_q6, conv4_q16, conv4_q22, conv4_q88, conv4_q95, conv4_q100, conv4_q109, conv4_q128, conv4_q144, conv4_q150, conv4_q153, conv4_q158, conv4_q166, conv5_q19, conv5_q23, conv5_q34, conv5_q37, conv5_q61, conv5_q77, conv5_q78, conv5_q81, conv5_q88, conv5_q115, conv6_q6, conv6_q17, conv6_q19, conv6_q53, conv6_q67, conv6_q73, conv6_q87, conv6_q94, conv6_q96, conv6_q103, conv6_q106, conv6_q117, conv6_q120, conv6_q130, conv6_q147, conv7_q34, conv7_q35, conv7_q45, conv7_q49, conv7_q50, conv7_q57, conv7_q61, conv7_q87, conv7_q93, conv7_q105, conv7_q106, conv7_q109, conv7_q118, conv7_q123, conv7_q145, conv7_q147, conv7_q156, conv7_q157, conv7_q169, conv7_q170, conv7_q184, conv7_q185, conv8_q15, conv8_q43, conv8_q61, conv8_q99, conv8_q133, conv8_q138, conv8_q142, conv8_q148, conv9_q0, conv9_q34, conv9_q40, conv9_q46, conv9_q88, conv9_q107, conv9_q109, conv9_q116`
- By category: `{'temporal': 22, 'multi-hop': 19, 'single-hop': 74, 'open-domain': 11}`
- Difference signals: `{'mag_much_slower_than_mem0': 126, 'mag_much_slower_than_zep': 126, 'mem0_answer_abstained': 37, 'zep_correct_but_answer_abstained': 13, 'mag_correct_but_answer_abstained': 3}`

- `conv0_q1` `temporal` signals=`mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: When did Melanie paint a sunrise?
  - GT: 2022
  - MAG: Melanie painted a lake sunrise in 2022.
  - Mem0: The memories do not indicate that Melanie painted a sunrise.
  - Zep: Melanie painted the sunrise in 2022.
  - support MAG/Mem0/Zep: `0.0/0.0/0.0`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv0_q9` `temporal` signals=`mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: When did Caroline meet up with her friends, family, and mentors?
  - GT: The week before 9 June 2023
  - MAG: Caroline met up with her friends, family, and mentors on June 9, 2023, and had a picnic with them on July 6, 2023.
  - Mem0: Caroline met up with her friends, family, and mentors on July 14, 2023 (council meeting for adoption), June 2023 (LGBT pride event), July 11, 2023 (joined 'Connected LGBTQ Activists'), and October 13, 2023 (contacted her mentor for adoption advice).
  - Zep: Caroline met up with her friends, family, and mentors on June 9, 2023.
  - support MAG/Mem0/Zep: `0.6/0.6/0.6`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv0_q34` `multi-hop` signals=`mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: What events has Caroline participated in to help children?
  - GT: Mentoring program, school speech
  - MAG: Caroline has participated in the following events to help children:
1. A school event where she encouraged students to engage with the LGBTQ community.
2. A council meeting for adoption that inspired her to adopt children.
3. Mentoring a transgender teen to build confidence and positive strategies.
4. An LGBT pride event that provided a supportive environment.
5. A mentorship program for LGBTQ youth.
6. An advocacy event promoting love and support for children and the LGBTQ community.
  - Mem0: Caroline participated in the following events to help children:
1. Attended a council meeting for adoption on July 14, 2023, which inspired her to adopt.
2. Received help from an adoption advice and assistance group while preparing for the adoption process.
  - Zep: Caroline has participated in the following events to help children:
1. Organizing a talent show for kids.
2. Joining a mentorship program for LGBTQ youth.
3. Attending an adoption advice/assistance group.
4. Participating in a council meeting for adoption.
  - support MAG/Mem0/Zep: `0.5/0.5/0.25`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv0_q40` `multi-hop` signals=`mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: How many times has Melanie gone to the beach in 2023?
  - GT: 2
  - MAG: 1-2 times in 2023 (likely once).
  - Mem0: 1
  - Zep: 2
  - support MAG/Mem0/Zep: `0.0/0.0/0.0`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv0_q43` `multi-hop` signals=`mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: What kind of art does Caroline make?
  - GT: abstract art
  - MAG: Caroline makes art that expresses her trans experience, promotes inclusivity and diversity, and serves as a means of self-expression and emotional exploration.
  - Mem0: Caroline makes paintings, particularly self-portraits, focusing on themes of inclusivity and diversity, advocating for the LGBTQ+ community, and exploring personal themes related to her transition and body image.
  - Zep: Caroline makes colorful paintings on canvas, including a self-portrait and a vibrant sunset painting. She also creates art that reflects her journey as a trans woman, using color symbolism to challenge rigid gender norms, and her work focuses on LGBTQ+ identity and acceptance.
  - support MAG/Mem0/Zep: `0.5/0.5/0.5`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv0_q56` `multi-hop` signals=`mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: What symbols are important to Caroline?
  - GT: Rainbow flag, transgender symbol
  - MAG: The important symbols to Caroline are the **rainbow flag**, which reflects the courage and strength of the trans community, and the **eagle**, which symbolizes freedom and pride.
  - Mem0: The important symbols to Caroline include:
1. A necklace with a cross and heart (symbolizing love, faith, and strength).
2. The painting 'Embracing Identity' (symbolizing comfort, love, and self-acceptance).
3. Colors red and blue (symbolizing the binary gender system).
4. Mixed colors (symbolizing breaking free from rigid thinking).
  - Zep: The important symbols to Caroline are:
1. A rainbow flag design, symbolizing togetherness and unity.
2. A pendant with a transgender symbol.
3. A necklace representing love, faith, and strength.
  - support MAG/Mem0/Zep: `1.0/0.25/0.75`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv0_q60` `multi-hop` signals=`mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: What instruments does Melanie play?
  - GT: clarinet and violin
  - MAG: Melanie plays the violin and the clarinet.
  - Mem0: Melanie does not play any instruments.
  - Zep: Melanie plays the clarinet.
  - support MAG/Mem0/Zep: `1.0/0.3333/0.6667`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv0_q103` `single-hop` signals=`mem0_answer_abstained,mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: What was Melanie's favorite book from her childhood?
  - GT: "Charlotte's Web"
  - MAG: Melanie's favorite book from her childhood was "Charlotte's Web."
  - Mem0: No information available about Melanie's favorite book from her childhood.
  - Zep: Melanie's favorite book from her childhood was "Charlotte's Web."
  - support MAG/Mem0/Zep: `1.0/0.0/1.0`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv0_q110` `single-hop` signals=`mem0_answer_abstained,mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: What kind of pot did Mel and her kids make with clay?
  - GT: a cup with a dog face on it
  - MAG: A cup.
  - Mem0: The specific kind of pot that Melanie and her kids made with clay is not mentioned in the memories.
  - Zep: Melanie's kids made a cup with a dog face during the pottery workshop.
  - support MAG/Mem0/Zep: `0.5/1.0/1.0`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv0_q135` `single-hop` signals=`mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: What setback did Melanie face in October 2023?
  - GT: She got hurt and had to take a break from pottery.
  - MAG: Melanie faced a setback in October 2023 when she got hurt and had to take a break from pottery.
  - Mem0: Melanie faced a setback when her son was involved in a car accident during a road trip on the weekend of October 14-15, 2023.
  - Zep: Melanie faced a setback in October 2023 due to an injury that required her to take a break from pottery, which she uses for self-expression and peace.
  - support MAG/Mem0/Zep: `0.8889/0.2222/0.7778`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv0_q138` `single-hop` signals=`zep_correct_but_answer_abstained,mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: What kind of painting did Caroline share with Melanie on October 13, 2023?
  - GT: An abstract painting with blue streaks on a wall.
  - MAG: Caroline shared that she has been trying out abstract painting.
  - Mem0: No specific painting was shared by Caroline on October 13, 2023.
  - Zep: The memories do not specify what kind of painting Caroline shared with Melanie on October 13, 2023.
  - support MAG/Mem0/Zep: `0.8333/0.5/0.6667`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv1_q8` `temporal` signals=`mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: When was Jon in Paris?
  - GT: 28 January 2023
  - MAG: Jon was in Paris on January 28, 2023.
  - Mem0: January 28, 2026.
  - Zep: Jon was in Paris on January 28, 2023.
  - support MAG/Mem0/Zep: `0.5/1.0/0.5`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv1_q26` `temporal` signals=`mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: When did Jon receive mentorship to promote his venture?
  - GT: 15 June, 2023
  - MAG: June 15, 2023.
  - Mem0: February 8, 2023.
  - Zep: Jon received mentorship to promote his venture on June 16, 2023.
  - support MAG/Mem0/Zep: `0.5/0.5/1.0`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv1_q68` `single-hop` signals=`mem0_answer_abstained,mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: What does Jon tell Gina he won't do?
  - GT: quit
  - MAG: Jon tells Gina he won't give up or quit.
  - Mem0: The memories do not specify what Jon tells Gina he won't do.
  - Zep: Jon tells Gina he won't give up or quit.
  - support MAG/Mem0/Zep: `1.0/0.0/1.0`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv2_q25` `multi-hop` signals=`mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: What European countries has Maria been to?
  - GT: Spain, England
  - MAG: Maria has been to Spain.
  - Mem0: Maria has not been recorded as visiting any European countries.
  - Zep: Maria has been to Spain.
  - support MAG/Mem0/Zep: `0.5/0.0/0.5`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv2_q27` `temporal` signals=`mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: When did John have a party with veterans?
  - GT: The Friday before 20 May 2023
  - MAG: May 20, 2023
  - Mem0: No record of John having a party with veterans.
  - Zep: May 20, 2023
  - support MAG/Mem0/Zep: `0.6/0.4/0.6`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv2_q32` `multi-hop` signals=`mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: What outdoor activities has John done with his colleagues?
  - GT: Hiking, mountaineering
  - MAG: John went on a hiking trip with his colleagues.
  - Mem0: John has not done any outdoor activities with his colleagues according to the memories.
  - Zep: John has gone on a hiking trip and a mountaineering trip with his colleagues.
  - support MAG/Mem0/Zep: `0.5/0.5/0.5`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv2_q73` `single-hop` signals=`mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: Who inspired Maria to start volunteering?
  - GT: Her aunt
  - MAG: Maria was inspired to start volunteering by her aunt, who believed in volunteering and had helped Maria's family when they were struggling.
  - Mem0: Maria was inspired to start volunteering by Jean, who taught her the importance of gratitude and connection.
  - Zep: Maria was inspired to start volunteering by her aunt, who believed in volunteering and helped her family when they were struggling.
  - support MAG/Mem0/Zep: `1.0/0.5/1.0`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv2_q84` `single-hop` signals=`mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: What did Maria participate in last weekend before April 10, 2023?
  - GT: a 5K charity run
  - MAG: Maria participated in a 5K charity run for a homeless shelter on the weekend of April 1-2, 2023.
  - Mem0: Maria participated in a charity event on February 3, 2023, where she served meals and felt the power of collective effort to help people in need.
  - Zep: Maria participated in a 5K charity run for a homeless shelter on April 7, 2023.
  - support MAG/Mem0/Zep: `1.0/0.5/1.0`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv2_q99` `single-hop` signals=`mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: What did John and the veterans do during the small party?
  - GT: share stories and make connections
  - MAG: John and the veterans shared their stories, fostering connections and camaraderie during the small party.
  - Mem0: No specific details about a small party involving John and veterans are available in the memories.
  - Zep: During the small party, John invited veterans to share their stories, and they made connections and found camaraderie, resulting in smiles and new friendships.
  - support MAG/Mem0/Zep: `1.0/0.6/1.0`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`
- Omitted examples: `106`; see CSV/JSONL outputs.

### Zep wrong, MAG/Mem0 correct

- Count: `103`
- Question IDs: `conv0_q3, conv0_q8, conv0_q16, conv0_q19, conv0_q21, conv0_q31, conv0_q45, conv0_q54, conv0_q57, conv0_q72, conv0_q84, conv0_q88, conv0_q104, conv1_q32, conv1_q40, conv1_q56, conv2_q5, conv2_q10, conv2_q13, conv2_q38, conv2_q43, conv2_q46, conv2_q63, conv2_q90, conv2_q91, conv2_q92, conv3_q22, conv3_q34, conv3_q39, conv3_q49, conv3_q79, conv3_q103, conv3_q110, conv3_q139, conv3_q150, conv4_q20, conv4_q31, conv4_q36, conv4_q60, conv4_q68, conv4_q96, conv4_q99, conv4_q107, conv4_q119, conv4_q135, conv5_q8, conv5_q63, conv5_q76, conv5_q87, conv5_q95, conv5_q96, conv5_q105, conv6_q18, conv6_q34, conv6_q48, conv6_q50, conv6_q71, conv6_q129, conv6_q138, conv6_q149, conv7_q9, conv7_q11, conv7_q12, conv7_q17, conv7_q23, conv7_q26, conv7_q28, conv7_q37, conv7_q46, conv7_q52, conv7_q62, conv7_q65, conv7_q77, conv7_q94, conv7_q103, conv7_q104, conv7_q114, conv7_q115, conv7_q116, conv7_q133, conv7_q139, conv7_q141, conv8_q9, conv8_q58, conv8_q63, conv8_q71, conv8_q75, conv8_q96, conv9_q5, conv9_q12, conv9_q30, conv9_q31, conv9_q35, conv9_q42, conv9_q53, conv9_q55, conv9_q70, conv9_q73, conv9_q75, conv9_q82, conv9_q92, conv9_q103, conv9_q134`
- By category: `{'multi-hop': 12, 'temporal': 43, 'single-hop': 43, 'open-domain': 5}`
- Difference signals: `{'zep_fact_context_not_sufficient': 103, 'mag_much_slower_than_mem0': 103, 'mag_much_slower_than_zep': 103, 'zep_answer_abstained': 22, 'mem0_correct_but_answer_abstained': 3, 'mag_correct_but_answer_abstained': 7}`

- `conv0_q3` `multi-hop` signals=`zep_fact_context_not_sufficient,mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: What did Caroline research?
  - GT: Adoption agencies
  - MAG: Caroline researched adoption agencies and mental health support for trans individuals.
  - Mem0: Caroline researched adoption agencies.
  - Zep: Caroline was going to do some research, likely related to job considerations, but the specific topic is not detailed.
  - support MAG/Mem0/Zep: `1.0/1.0/1.0`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv0_q8` `temporal` signals=`zep_fact_context_not_sufficient,mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: When did Caroline give a speech at a school?
  - GT: The week before 9 June 2023
  - MAG: Caroline gave a speech at a school event during the week of June 2-8, 2023.
  - Mem0: Caroline gave a speech at a school event on June 2-3, 2023.
  - Zep: No record of Caroline giving a speech at a school.
  - support MAG/Mem0/Zep: `0.6/0.6/0.6`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv0_q16` `temporal` signals=`zep_fact_context_not_sufficient,mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: When did Melanie sign up for a pottery class?
  - GT: 2 July 2023
  - MAG: Melanie signed up for a pottery class on July 2, 2023.
  - Mem0: July 2, 2023.
  - Zep: Melanie signed up for a pottery class on July 3, 2023.
  - support MAG/Mem0/Zep: `0.5/1.0/0.5`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv0_q19` `multi-hop` signals=`zep_fact_context_not_sufficient,mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: What do Melanie's kids like?
  - GT: dinosaurs, nature
  - MAG: Melanie's kids like outdoor activities, particularly exploring and playing at the park.
  - Mem0: Melanie's kids like painting, pottery, learning about nature, and playing at the beach with a kite.
  - Zep: Melanie's kids like seeing sand castles at the beach, making pottery (like a cup with a dog face), exploring and playing at the park, and going camping.
  - support MAG/Mem0/Zep: `0.0/0.5/0.0`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv0_q21` `temporal` signals=`zep_fact_context_not_sufficient,mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: When did Caroline have a picnic?
  - GT: The week before 6 July 2023
  - MAG: Caroline had a picnic during the week of June 26 to July 2, 2023.
  - Mem0: Caroline had a picnic with friends around the week of June 30, 2023.
  - Zep: Caroline had a picnic last week, specifically during the week of 15-21 October 2023.
  - support MAG/Mem0/Zep: `0.8/0.6/0.6`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv0_q31` `temporal` signals=`zep_fact_context_not_sufficient,mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: When did Melanie go camping in June?
  - GT: The week before 27 June 2023
  - MAG: Melanie went camping in the mountains during the week of June 19-25, 2023.
  - Mem0: Melanie went camping in the mountains around June 20, 2023.
  - Zep: No specific camping trip in June is mentioned for Melanie.
  - support MAG/Mem0/Zep: `0.6/0.8/0.4`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv0_q45` `temporal` signals=`zep_fact_context_not_sufficient,mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: When did Caroline attend a pride parade in August?
  - GT: The Friday before 14 August 2023
  - MAG: Caroline attended a pride parade on August 11, 2023.
  - Mem0: Caroline attended a pride parade on August 11, 2023.
  - Zep: Caroline attended a pride parade on August 14, 2023.
  - support MAG/Mem0/Zep: `0.6/0.8/0.8`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv0_q54` `temporal` signals=`zep_fact_context_not_sufficient,mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: When did Caroline draw a self-portrait?
  - GT: The week before 23 August 2023
  - MAG: Caroline drew a self-portrait during the week of 16 to 23 August 2023.
  - Mem0: Caroline drew a self-portrait on August 16, 2023.
  - Zep: Caroline drew a self-portrait during the week of 15-21 October 2023.
  - support MAG/Mem0/Zep: `0.6/0.8/0.8`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv0_q57` `temporal` signals=`zep_fact_context_not_sufficient,mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: When did Caroline encounter people on a hike and have a negative experience?
  - GT: The week before 25 August 2023
  - MAG: Caroline encountered people on a hike and had a negative experience around August 18-24, 2023.
  - Mem0: Caroline encountered people on a hike and had a negative experience around August 18, 2023.
  - Zep: Caroline encountered people on a hike and had a negative experience during the week of October 15, 2023.
  - support MAG/Mem0/Zep: `0.6/0.8/0.8`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv0_q72` `temporal` signals=`zep_fact_context_not_sufficient,mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: When did Melanie's friend adopt a child?
  - GT: 2022
  - MAG: Melanie's friend adopted a child in 2022.
  - Mem0: Melanie's friend adopted a child in 2022.
  - Zep: Caroline has not yet adopted a child; she applied to adoption agencies on August 23, 2023.
  - support MAG/Mem0/Zep: `0.0/0.0/0.0`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv0_q84` `single-hop` signals=`zep_fact_context_not_sufficient,mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: How does Melanie prioritize self-care?
  - GT: by carving out some me-time each day for activities like running, reading, or playing the violin
  - MAG: Melanie prioritizes self-care by engaging in activities like swimming with her kids and recognizing that taking care of herself is essential for her ability to care for her family.
  - Mem0: Melanie prioritizes self-care by participating in a charity race for mental health, using pottery for self-expression and peace, and engaging in painting and reading during her break from pottery.
  - Zep: Melanie prioritizes self-care by engaging in pottery for self-expression and peace, creating calming art, and recognizing that taking care of herself enables her to better support her family.
  - support MAG/Mem0/Zep: `0.2143/0.2857/0.2857`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv0_q88` `single-hop` signals=`zep_fact_context_not_sufficient,mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: What is Caroline excited about in the adoption process?
  - GT: creating a family for kids who need one
  - MAG: Caroline is excited about researching adoption agencies, the emotional inspiration from attending a council meeting, applying to adoption agencies, passing the adoption agency interviews, and the opportunity to provide a loving home for children in need.
  - Mem0: Caroline is excited about her future and being on an adoption journey, starting a new chapter in her life as she is ready to be a mom, and passing the adoption agency interviews, which is a significant step towards her goal of having a family.
  - Zep: Caroline is excited about passing the adoption agency interviews, which she accomplished on October 20, 2023.
  - support MAG/Mem0/Zep: `1.0/0.2857/0.4286`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv0_q104` `single-hop` signals=`zep_answer_abstained,zep_fact_context_not_sufficient,mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: What book did Caroline recommend to Melanie?
  - GT: "Becoming Nicole"
  - MAG: Caroline recommended "Becoming Nicole" by Amy Ellis Nutt to Melanie.
  - Mem0: Caroline recommended the book 'Becoming Nicole' by Amy Ellis Nutt to Melanie.
  - Zep: The specific title of the book that Caroline recommended to Melanie is not mentioned in the memories.
  - support MAG/Mem0/Zep: `1.0/1.0/0.0`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv1_q32` `temporal` signals=`zep_fact_context_not_sufficient,mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: When did Gina design a limited collection of hoodies?
  - GT: June 2023
  - MAG: Gina designed a limited collection of hoodies during the week of June 14, 2023.
  - Mem0: Gina designed a limited collection of hoodies around June 14-20, 2023.
  - Zep: July 16, 2023.
  - support MAG/Mem0/Zep: `0.5/1.0/1.0`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv1_q40` `single-hop` signals=`mem0_correct_but_answer_abstained,zep_answer_abstained,zep_fact_context_not_sufficient,mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: What is Jon's favorite style of dance?
  - GT: Contemporary
  - MAG: Jon's favorite style of dance is contemporary.
  - Mem0: Jon's favorite style of dance is not mentioned in the memories.
  - Zep: Jon's favorite style of dance is not mentioned in the memories.
  - support MAG/Mem0/Zep: `1.0/0.0/0.0`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv1_q56` `single-hop` signals=`zep_fact_context_not_sufficient,mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: What did Jon and Gina compare their entrepreneurial journeys to?
  - GT: dancing together and supporting each other
  - MAG: Jon and Gina compared their entrepreneurial journeys to dancing together.
  - Mem0: Jon and Gina compared their entrepreneurial journeys to having a partner to dance with.
  - Zep: Jon and Gina did not explicitly compare their entrepreneurial journeys to anything specific in the available memories.
  - support MAG/Mem0/Zep: `0.5/0.8333/0.3333`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv2_q5` `temporal` signals=`zep_fact_context_not_sufficient,mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: When did Maria go to the beach?
  - GT: December 2022
  - MAG: December 2022
  - Mem0: Maria went to the beach during her solo trip to Spain in 2022.
  - Zep: Maria went to the beach in July 2023.
  - support MAG/Mem0/Zep: `0.0/0.0/0.0`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv2_q10` `temporal` signals=`zep_fact_context_not_sufficient,mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: When did Maria meet Jean?
  - GT: February 24, 2023
  - MAG: Maria met Jean on February 25, 2023.
  - Mem0: February 24, 2023.
  - Zep: Maria met Jean while volunteering on or after February 25, 2023.
  - support MAG/Mem0/Zep: `0.5/1.0/0.5`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv2_q13` `temporal` signals=`zep_fact_context_not_sufficient,mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: When did Maria's grandmother pass away?
  - GT: The week before 6 March 2023
  - MAG: Maria's grandmother passed away in late February 2023.
  - Mem0: Maria's grandmother passed away on March 1, 2023.
  - Zep: Maria's grandmother passed away on or around 9 August 2023.
  - support MAG/Mem0/Zep: `0.6/0.6/0.4`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`

- `conv2_q38` `temporal` signals=`zep_fact_context_not_sufficient,mag_much_slower_than_mem0,mag_much_slower_than_zep`
  - Q: When did Maria get in a car accident?
  - GT: July 2, 2023
  - MAG: July 3, 2023.
  - Mem0: Maria got in a car accident on July 2, 2023.
  - Zep: Maria got in a car accident on July 3, 2023.
  - support MAG/Mem0/Zep: `0.5/1.0/0.5`; overlap MAG-Mem0/MAG-Zep: `0.0/0.0`
- Omitted examples: `83`; see CSV/JSONL outputs.

## Difference Signal Counts

| Signal | Count |
|---|---:|
| `mag_much_slower_than_mem0` | 1531 |
| `mag_much_slower_than_zep` | 1530 |
| `zep_fact_context_not_sufficient` | 230 |
| `mag_low_topk_overlap_with_mem0` | 145 |
| `mag_low_topk_overlap_with_zep` | 134 |
| `zep_temporal_fact_context_available` | 134 |
| `mag_graph_heavy_when_wrong` | 130 |
| `mag_gt_support_lower_than_correct_baselines` | 107 |
| `zep_correct_but_answer_abstained` | 70 |
| `mem0_answer_abstained` | 58 |
| `mag_answer_abstained` | 54 |
| `zep_answer_abstained` | 47 |
| `mag_correct_but_answer_abstained` | 39 |
| `mem0_correct_but_answer_abstained` | 33 |
