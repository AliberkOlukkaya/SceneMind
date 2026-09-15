"""Build the frozen, manually reviewed human-grounded routing manifest.

Every event below was written after reviewing the extracted five-second contact
sheets and the local Whisper transcript. Queries are literal annotations rather
than generated paraphrases; this file makes that review trail reproducible.
"""

from __future__ import annotations

import json
from pathlib import Path

EVENTS = {
    "good-sources-explainer": [
        (5, 15, "Wikipedia page and animated reader are visible", "the narrator introduces kinds of Wikipedia information", "Wikipedia page with a cartoon reader", "What kinds of information does the narrator say can be found on Wikipedia?", "Find the Wikipedia page on screen while the narrator introduces the kinds of information it contains"),
        (15, 25, "several volunteer figures appear around an article", "Wikipedia articles are said to be written by volunteer authors", "the group of volunteers around an article", "Who does the speaker say writes Wikipedia articles?", "Where are volunteer figures shown as the narrator explains who contributes articles?"),
        (25, 38, "a crowd and religious and political symbols depict conflict", "the narration discusses conflicting opinions and neutral point of view", "crowd arguing under political and religious symbols", "Why does the narrator say articles need a neutral point of view?", "Find the conflicting crowd shown while neutrality is being explained"),
        (39, 55, "thought bubbles and competing theory labels appear", "the narrator says an article should list theories without endorsing them", "two competing theory bubbles", "How should an article handle theories according to the explanation?", "Show the competing theory bubbles when the voice says not to accept or reject either one"),
        (56, 70, "a scale and books represent credibility and sources", "the narration says authors demonstrate knowledge through sources", "balance scale next to a stack of books", "How can an author show knowledge and credibility?", "Locate the scales and books while sources are described as evidence of credibility"),
        (70, 82, "Anna, the Moon, and books are visible", "the example says Anna's Moon claim needs a source", "cartoon woman pointing toward the Moon", "What claim by Anna does the narrator say needs a source?", "Find Anna beside the Moon when her claim is said to require a citation"),
        (82, 96, "academic books and a personal blog page are contrasted", "the voice distinguishes trustworthy sources from personal blogs", "books beside a blog webpage", "Why is a personal blog rejected as a reliable source?", "Show the blog page while the narrator contrasts it with academic and press sources"),
        (96, 106, "two Wikipedia pages point at each other in a loop", "the narrator calls citing Wikipedia from Wikipedia a vicious circle", "two linked web pages forming a circle", "What does the narrator call using Wikipedia itself as a citation?", "Find the circular page links when the speaker warns about a vicious circle"),
        (106, 118, "link and reliability symbols appear around pages", "the voice explains that links help readers verify reliability", "web pages connected by chain links", "What benefit of links does the narrator mention?", "Locate the linked pages while the narrator explains how readers check reliability"),
        (118, 128, "a round Earth rotates on screen", "the narrator says established facts do not need citations", "the rotating planet Earth", "Which kind of facts does the speaker say need no citation?", "Find the globe when the narrator gives a round Earth as an established fact"),
    ],
    "oceans-explainer": [
        (5, 28, "ocean water, a turtle, and mangrove roots are visible", "mangroves are introduced as salt-water forests", "sea turtle swimming past mangrove roots", "How are mangrove forests described at the start?", "Find the turtle near mangrove roots while their salt-water habitat is introduced"),
        (35, 55, "small fish shelter among mangrove roots", "the narrator says the forests protect young fish and coasts", "school of juvenile fish among tree roots", "What protection do mangrove forests provide?", "Show the young fish in roots as the narrator explains the nursery habitat"),
        (58, 88, "bright coral reef and many fish fill the frame", "coral polyps and algae are described as feeding one another", "colorful coral reef crowded with fish", "How do coral polyps and algae help each other?", "Locate the colorful reef while the symbiosis between polyps and algae is explained"),
        (98, 125, "a sea turtle crosses open blue water", "the narration describes a migration from Mexico toward Japan", "lone turtle swimming through open ocean", "Where does the turtle travel to lay her eggs?", "Find the migrating turtle while the narrator describes her Pacific journey"),
        (130, 158, "plankton-like organisms and broad ocean imagery appear", "the voice says more than half of oxygen comes from plankton", "tiny drifting organisms in blue water", "What produces more than half of our oxygen?", "Show the drifting plankton imagery as its oxygen contribution is stated"),
        (180, 205, "a glowing deep-sea fish uses a lure in darkness", "the narrator explains its lantern contains luminous bacteria", "deep-sea fish with a glowing lure", "What makes the female frogfish's lantern glow?", "Find the glowing lure when the narrator explains how it attracts prey"),
        (205, 232, "black smokers vent from the seafloor", "the narration describes toxic water above 660 degrees Fahrenheit", "dark hydrothermal vents on the ocean floor", "How hot does the narrator say the vent water can be?", "Locate the black vents while their toxic extreme heat is described"),
        (285, 318, "plastic bottles and fragments drift underwater", "eight million metric tons of garbage are said to enter the sea yearly", "plastic waste floating among marine life", "How much garbage does the speaker say reaches the ocean each year?", "Show the plastic debris when the annual pollution quantity is given"),
        (330, 360, "hooks, nets, and discarded fish depict bycatch", "the narrator says shrimp catch can discard nine pounds of other life", "dead fish caught beside fishing gear", "What bycatch ratio is mentioned for a pound of shrimp?", "Find the discarded fish while the narrator quantifies shrimp bycatch"),
        (412, 445, "a reef loses color and turns white", "warming is explained as causing polyps to expel algae", "coral changing from colorful to white", "Why does warm water make coral turn white?", "Show the bleaching reef while the narrator explains the expelled algae"),
    ],
    "webm-codec-explainer": [
        (6, 18, "a laptop plays video beside a codec timeline", "the narrator introduces the history of video codecs", "laptop next to a historical timeline", "What history does the speaker say the video will cover?", "Find the codec timeline when the narrator introduces video on the web"),
        (18, 32, "the WebM logo and browser video are visible", "WebM is described as optimized for Internet viewing", "WebM logo beside an online video player", "What environment was WebM optimized for?", "Show the WebM logo while its web-viewing purpose is explained"),
        (32, 46, "teleconference and broadcast television images appear", "older codecs are said to come from teleconferencing and TV", "video conference beside an old television", "Where did many older video codecs originate?", "Locate the TV and conference images as their codec history is narrated"),
        (47, 61, "one broadcast antenna feeds many televisions", "broadcasting is explained as sending a single stream", "antenna distributing one signal to many TVs", "How many streams does traditional broadcast send?", "Find the antenna diagram while the narrator defines a broadcast stream"),
        (62, 78, "individual laptops connect separately to a server", "web video is described as unicast", "server sending separate lines to laptops", "What does unicast mean in the explanation?", "Show the separate laptop connections while unicast delivery is described"),
        (78, 90, "many laptop icons multiply around a server", "a thousand viewers are said to require a thousand streams", "crowd of computers connected to one server", "How many streams are needed for a thousand web viewers?", "Find the many-computer diagram when the narrator gives the thousand-user example"),
        (90, 103, "phones, tablets, laptops, and TVs fill the screen", "the narrator lists varying browsers, resolutions, and connections", "collection of different playback devices", "What viewer differences must a web codec handle?", "Show the mixed devices while different browsers and resolutions are discussed"),
        (103, 114, "older and newer screens are contrasted", "the speaker says these constraints require different technology", "old television facing a modern device", "Why does web video need different technology?", "Locate the old and new displays as the special web constraints are summarized"),
        (114, 122, "a large WebM wordmark is centered", "the project goal is summarized as a codec for the web", "large WebM logo on a plain background", "What goal does the narrator give for the WebM project?", "Find the centered WebM mark while the project's goal is stated"),
        (121, 126, "the final WebM card includes an invitation", "listeners are invited to get involved with open source", "WebM end card", "What invitation closes the video?", "Show the final WebM card as viewers are invited to join the open-source project"),
    ],
    "blended-learning-explainer": [
        (5, 28, "online and classroom icons combine in a diagram", "blended learning is defined as online resources plus in-person instruction", "diagram joining a laptop with a classroom", "How does the narrator define blended learning?", "Find the combined laptop-classroom diagram while blended learning is defined"),
        (28, 48, "home lecture and classroom assignment scenes swap places", "the flipped classroom model is explained", "student watching a lecture at home", "What do students do at home in a flipped classroom?", "Show the home lecture scene while the narrator explains the flipped model"),
        (50, 70, "difficulty branches show bored and lost students", "teachers are said to lecture to a mythical middle", "forked diagram with bored and confused students", "Why do teachers aim at the mythical middle?", "Locate the difficulty branches when the mythical middle problem is described"),
        (72, 102, "video controls show speed, pause, and rewind", "students can move at their own pace and call up resources", "online lecture player with playback controls", "What pacing choices can students make with online content?", "Find the pause and rewind controls as self-paced study is explained"),
        (112, 142, "students work at desks with a teacher nearby", "active work is said to be better than passive listening", "teacher helping students with assignments", "Why put work assignments back into class time?", "Show the teacher helping at desks while active learning benefits are discussed"),
        (148, 174, "a classroom group gathers around a table", "in-class group work simplifies logistics and facilitation", "students collaborating around one table", "How does classroom group work solve coordination problems?", "Find the group around a table while the logistical benefit is explained"),
        (174, 202, "online-only and face-to-face layouts are contrasted", "blended classes are said to preserve classroom interaction", "split comparison of online and classroom interaction", "Does blended learning remove face-to-face time?", "Show the two learning layouts while the narrator distinguishes blended from online classes"),
        (212, 258, "a time-based track changes to individual mastery paths", "students may move on only after demonstrating mastery", "parallel student progress tracks at different speeds", "When do students move forward in a mastery-based model?", "Locate the different progress tracks as mastery-based pacing is explained"),
        (266, 300, "warning icons list cost, reliability, and learning curve", "technical resources must be affordable, reliable, and usable", "three challenge icons beside a computer", "What technical requirements can challenge blended learning?", "Find the challenge icons while affordability and reliability are listed"),
        (302, 334, "teacher shifts beside a student in the driver's seat", "the role changes from sage on the stage to guide on the side", "student steering while teacher coaches", "How does the teacher's role change?", "Show the student in control as the narrator describes the teacher becoming a coach"),
    ],
    "internet-language-talk": [
        (20, 35, "title slide and remote presenter are visible", "the speaker proposes changing the Internet's default language", "speaker beside the talk title slide", "What change does the presenter propose for the Internet?", "Find the title slide while the speaker states her proposal"),
        (36, 52, "a slide displays the code de", "the speaker asks what language code German uses", "large lowercase de on a dark slide", "Which language is represented by the code de?", "Show the de slide as the presenter identifies the German language code"),
        (50, 72, "slides change from de-DE to de-AT", "German and Austrian regional variants are contrasted", "de-DE followed by de-AT on screen", "What regional distinction does the speaker make between de-DE and de-AT?", "Locate the de-AT slide while Austrian German is contrasted with German German"),
        (74, 98, "a syntax diagram labels language code and country code", "IETF BCP 47 language tags are explained", "language-tag diagram with two code parts", "What standard for language tags does the presenter name?", "Find the two-part code diagram as BCP 47 is explained"),
        (98, 115, "the slide shifts to locale terminology", "localization is said to cover more than language", "locale label on the presentation slide", "Why does the speaker say localization is more than language?", "Show the locale slide while its broader meaning is described"),
        (118, 150, "en and en-US occupy consecutive slides", "English defaults are described as implicitly American", "en-US shown beneath a large en code", "What national convention often hides behind default English?", "Find en-US on screen as the speaker discusses American defaults"),
        (145, 172, "an American date example is displayed", "the presenter calls the month-first format unusual outside the US", "month-day-year date printed below en-US", "What date-format problem does the speaker identify?", "Show the en-US date example while its ordering is criticized"),
        (174, 206, "the Unicode Common Locale Data Repository title appears", "CLDR is described as supplying locale data to operating systems and browsers", "Unicode CLDR title slide", "What does CLDR provide to browsers and operating systems?", "Locate the CLDR title while the presenter explains what systems obtain from it"),
        (208, 240, "en-150 and a data-deduplication meme appear", "European English and its parent relationship are discussed", "en-150 slide followed by a duplication meme", "When was European English introduced into CLDR?", "Find the en-150 slide while its history and parent locale are explained"),
        (254, 303, "en-001 changes color beside a world-scale bar", "international English is recommended as the best world default", "orange en-001 code with a global scale", "Which locale does the presenter recommend as the Internet default?", "Show the orange en-001 slide while it is recommended as international English"),
    ],
    "design-free-software-talk": [
        (18, 42, "conference title slide and remote speaker appear", "the speaker identifies herself and her universities", "remote presenter beside the Libre Graphics title", "Where does Lila Pagola say she teaches?", "Find the conference title slide while the presenter introduces her affiliations"),
        (60, 92, "a slide asks how much and which software to teach", "software is described as design's primary environment", "question slide about software in design education", "Why does the presenter call software more than a tool for designers?", "Show the education question slide while software's role is explained"),
        (125, 170, "assignment steps are listed in colored text", "students must identify, test, and publicly review a free tool", "slide listing the free-software assignment", "What are students required to do in the assignment?", "Locate the assignment checklist while the presenter explains the practical review task"),
        (230, 285, "a black-box theory slide shows hardware and software questions", "the speaker says tools contain encoded developer decisions", "two-level black-box diagram", "What does the black-box analogy reveal about software?", "Find the black-box slide while encoded decisions inside tools are discussed"),
        (300, 340, "free-software principles and case studies are listed", "Blender is cited to show free does not mean limited", "slide naming Blender among case studies", "Why does the presenter mention Blender?", "Show Blender in the case-study list as the speaker argues that free software is capable"),
        (330, 380, "two columns contrast unfamiliarity and usability gaps", "students are asked to separate frustration from objective defects", "comparison slide with two kinds of discomfort", "How should students distinguish unfamiliarity from a software problem?", "Locate the two-column slide while fair tool criticism is explained"),
        (390, 445, "a diagram lists blind trust and recurring review themes", "reviews revealed weak understanding of software types", "review-findings slide with several recurring themes", "What problem did the student reviews reveal?", "Find the recurring-themes diagram while the pedagogical findings are presented"),
        (445, 485, "student blog screenshots are shown", "the speaker says technical training should restore creative agency", "screenshots of students' software reviews", "What kind of agency does free software education aim to restore?", "Show the student review pages while agency over creative tools is discussed"),
        (490, 540, "quoted student comments fill the slide", "students report that alternatives changed how they understood familiar tools", "two blocks of student feedback", "What did students learn from testing alternative software?", "Find the feedback quotes while the testing process is described as the main outcome"),
        (1180, 1260, "the remote speaker remains on camera during Q&A", "she argues public-university students need exposure to alternatives", "presenter speaking from a room lined with shelves", "Why does she consider free-software exposure a public-university responsibility?", "Show the presenter on camera while she defends teaching critical alternatives"),
    ],
    "homemade-pasta-instruction": [
        (10, 30, "host stands behind flour and bowls", "she introduces eggless pasta and semolina flour", "cook beside bowls of flour", "What flour does Laura introduce for eggless pasta?", "Find the ingredient bowls while she introduces semolina pasta flour"),
        (35, 65, "two flours are measured and combined", "the recipe calls for equal parts all-purpose and pasta flour", "measuring cups tipping flour into a bowl", "What ratio of the two flours does the recipe use?", "Show the flour being measured while Laura specifies equal parts"),
        (68, 98, "a well is formed in flour and filled with water", "three quarters of a cup of cold water is recommended initially", "water poured into a well in the flour", "How much cold water should be added first?", "Find the flour well as Laura gives the starting water amount"),
        (125, 165, "dough is pressed and folded by hand", "kneading for about five minutes is instructed", "hands kneading dough on a silicone mat", "How long should the dough be kneaded?", "Show the kneading motion while the five-minute instruction is given"),
        (165, 190, "the dough is wrapped in plastic", "it rests for half an hour so the gluten loosens", "ball of dough covered in plastic wrap", "Why should the dough rest for half an hour?", "Find the wrapped dough while Laura explains that the gluten must relax"),
        (205, 240, "a bench scraper divides dough and pieces become logs", "working with one small section at a time is advised", "bench scraper cutting the dough into quarters", "Why does she divide the dough into small sections?", "Show the scraper cutting dough while she explains working one section at a time"),
        (242, 282, "thumbs drag small pieces into shell shapes", "the hand-shaped pasta is named orecchiette", "small pasta pieces shaped with a thumb", "What is the thumb-shaped pasta called?", "Find the thumb-dragging step when Laura names the orecchiette shape"),
        (305, 345, "a leek is sliced and washed in a bowl", "dirt between leek layers is said to sink in still water", "chopped leek pieces soaking in water", "Why should the leek sit without being stirred?", "Show the leek in water while the dirt-settling method is explained"),
        (435, 470, "a lemon is squeezed over a glass bowl", "a heavy lemon is said to contain more juice", "yellow lemon in a handheld juicer", "How can you tell that a lemon has plenty of juice?", "Find the lemon juicer while Laura explains why a heavy lemon is preferable"),
        (590, 630, "pasta boils and pieces float to the surface", "floating after seven to eight minutes signals doneness", "orecchiette floating in a pot", "How does Laura know the pasta is cooked?", "Show the boiling pot while she says the pieces float when ready"),
    ],
    "human-software-extensions-talk": [
        (65, 92, "film stills show a plugged-in worker and industrial robots", "Sleep Dealer is described as replacing US workers with remotely controlled robots", "worker with neck cables followed by factory robots", "What future does the film Sleep Dealer imagine?", "Find the plugged-in worker while the speaker explains who controls the robots"),
        (145, 188, "a layered stack diagram runs from user to Earth", "human software extensions are defined as bodies and cognition plugged into systems", "stack of layers beside a wireframe globe", "How does the speaker define humans as software extensions?", "Show the planetary stack diagram while bodies and minds are described as system components"),
        (238, 290, "purple Earth and a workstation montage appear", "the stack is called both reality and an ideology of optimization", "purple globe followed by a multi-monitor workstation", "Why does the speaker call planetary computation an ideology as well as a reality?", "Find the purple globe while the scale and power of the stack are discussed"),
        (310, 350, "Upwork profile and contract controls are visible", "freelancers can be sorted by price, skills, and rating", "freelancer marketplace with Pay and End Contract buttons", "How does Upwork let clients choose freelancers?", "Show the contract controls while the hire-and-fire interface is described"),
        (350, 390, "an activity diary displays keyboard and mouse counts", "Upwork is said to record keystrokes and take screenshots", "work log showing activity level and input events", "What monitoring does Upwork perform on freelancers?", "Find the activity diary while the speaker explains how clients can spy on workers"),
        (455, 495, "Amazon Key scenes show a phone and apartment access", "the service is said to let users grant access without meeting workers", "person using a phone outside an apartment", "What does Amazon Key allow someone to do remotely?", "Show the apartment-access app while invisible workers are discussed"),
        (535, 590, "Fiverr marketplace tiles and a large collage fill the screen", "five-dollar gigs and platform competition are explained", "dense collage of Fiverr gig thumbnails", "How was Fiverr's original five-dollar price divided?", "Find the gig collage while the presenter explains the platform's original fee"),
        (615, 670, "logo animations and performers appear in a four-panel montage", "workers use bots and templates to meet unrealistic delivery times", "four-panel grid of animated logo gigs", "Why do some gig workers automate their creative work?", "Show the logo-demo grid while short delivery times and templates are discussed"),
        (795, 855, "hundreds of outlined objects flicker through segmentation masks", "the masks are said to be manually made by Mechanical Turk workers for COCO", "rapid sequence of hand-drawn object outlines", "Who created the segmentations used for the COCO recognition dataset?", "Find the flashing outline masks while their hidden manual labor is explained"),
        (930, 1005, "handmade accordion books unfold into a long strip", "five years of CAPTCHA micro-labor are said to span ninety meters", "hand unfolding a tall concertina book", "What does the ninety-meter CAPTCHA collection chronicle?", "Show the accordion books while the speaker describes five years of micro-labor"),
    ],
}

# Additional examples deliberately cross query form and route: each class contains
# questions, commands, noun phrases, and indirect requests. These were added before
# the two final-test sources were evaluated.
EVENTS["good-sources-explainer"].extend([
    (13, 23, "five author figures surround a Wikipedia article", "the narration says everyone can contribute", "Who is visible around the article page?", "Wikipedia's volunteer authors", "Which group surrounds the page when the narrator says anyone may contribute?"),
    (31, 47, "opposing symbols and speech bubbles appear", "neutral presentation of disputes is required", "Can you take me to the picture with opposing political and religious symbols?", "Find the explanation of neutral treatment of disputes", "What symbols are displayed as neutral point of view is defined?"),
    (58, 72, "a knowledge scale gives way to Anna and the Moon", "claims need evidence from credible publications", "the woman beside a crescent moon", "When are citations required for a claim?", "I remember a Moon drawing during an example about evidence—where was that?"),
    (83, 101, "a blog page is followed by circular Wikipedia links", "personal blogs and circular citations are rejected", "What kind of page is shown before the two circular links?", "unreliable personal sources and circular citation", "Take me to the blog graphic during the warning about unreliable sources"),
    (113, 128, "the connected pages change into a globe", "established facts are exempted from citation", "At what point does Earth fill the frame?", "established facts needing no citation", "Which familiar fact is pictured when the exception to citation is described?"),
])
EVENTS["oceans-explainer"].extend([
    (145, 175, "wide ocean and climate imagery appear", "oceans store heat and absorb far more greenhouse gas than air", "What is on screen during the move from the surface into deep water?", "ocean heat storage and greenhouse gas absorption", "Find the broad ocean view accompanying the claim about climate and stored heat"),
    (235, 270, "dark water and unknown silhouettes suggest unexplored depths", "humans know less about oceans than the Moon's far side", "the dark mystery-of-the-depths sequence", "How little do humans know about the ocean?", "Which mysterious deep-sea images accompany the comparison with the Moon?"),
    (315, 342, "microplastic gives way to falling fish populations", "the prediction says plastic may outnumber fish by 2050", "Show me the transition from plastic fragments to fish", "more plastic than fish by 2050", "What imagery appears when the 2050 plastic prediction is spoken?"),
    (365, 402, "murky runoff and factory discharge enter water", "fertilizer, sewage, and chemicals create dead zones", "Where can I see dirty water flowing toward the sea?", "What causes marine dead zones?", "Find the runoff scene paired with the explanation of falling oxygen levels"),
    (445, 474, "waves strike an exposed coast and a turtle appears", "sustainable treatment is tied to human health", "coastline without mangrove protection", "our health depends on ocean health", "What unprotected shore is shown as the closing sustainability message is delivered?"),
])
EVENTS["webm-codec-explainer"].extend([
    (10, 29, "timeline marks the move toward web video", "old codecs were not designed for browser delivery", "Which timeline is visible near the beginning?", "Why was a new web codec needed?", "Find the history line behind the explanation that older codecs predate web viewing"),
    (38, 57, "televisions receive one common signal", "broadcast delivery is contrasted with requests on the web", "a row of televisions under one antenna", "broadcast versus requested video streams", "What shared-TV diagram is used while broadcast delivery is contrasted with the web?"),
    (57, 78, "one server sends individual lines to computers", "each web viewer receives an independent unicast stream", "Where is the purple server with separate laptop connections?", "Explain unicast in this video", "Which network picture accompanies the point that every viewer requests a stream?"),
    (86, 106, "many device types and screens are shown", "playback conditions differ by browser, resolution, and bandwidth", "What devices appear together in the crowded frame?", "different browsers, resolutions, and connections", "Take me to the device collage during the list of varied playback conditions"),
    (108, 125, "the WebM mark dominates the closing frames", "WebM's open-source goal and invitation close the video", "the large WebM wordmark", "How can viewers participate in the project?", "When does the WebM logo appear as the open-source invitation is made?"),
])
EVENTS["blended-learning-explainer"].extend([
    (92, 116, "early-bird and night-owl icons appear with accessibility breaks", "learners can study when alert and pause for disability needs", "Which two bird-themed learner icons are displayed?", "benefits for disabled students taking breaks", "What icons appear while flexible study time and needed breaks are discussed?"),
    (185, 210, "teacher moves from a stage to individual student help", "face time is optimized into one-to-one guidance", "Show the instructor leaving the lecture stage", "guide on the side instead of sage on the stage", "Where does the teacher approach a student as the new coaching role is named?"),
    (220, 248, "equal class tracks split into differently paced paths", "the whole class no longer advances on a fixed date", "How do the progress tracks change on screen?", "Why can mastery-based students advance at different times?", "Find the diverging tracks during the contrast with fixed-time lessons"),
    (278, 306, "technology warnings lead into linked assignment icons", "preparation can be encouraged by tying class work to online material", "the warning signs and connected assignment symbols", "How can teachers encourage students to arrive prepared?", "Which linked-work graphic appears when preparation incentives are explained?"),
    (330, 362, "recap diagrams revisit laptop, classroom, and coach", "the summary repeats flipped classroom and student focus", "What three learning pictures return in the recap?", "quick recap of the flipped classroom", "Show the final recap graphics while the narrator summarizes student-focused learning"),
])

EVENTS.update({
    "rewiring-video-editor-talk": [
        (40, 82, "a traditional timeline fills with colored clips", "the speaker calls layer-based editing archaic and describes rearranging clips", "Why are the colored blocks being moved around the timeline?", "traditional NLE editing is archaic", "Show me the crowded colored timeline while the presenter explains the three-step clip move"),
        (105, 140, "blue audio is visibly cut beneath rearranged clips", "adjacent movement can destructively cut linked audio", "the timeline where a blue strip gets sliced", "What can happen to linked audio when a clip moves?", "Which blue track is cut as destructive editing is described?"),
        (180, 225, "a time node drives a cube, then keyframe graphs appear", "the simplest node level cannot support keyframing", "Can I see the cube connected to the time node?", "why the simplest time-node design is inflexible", "Find the moving cube example where the lack of keyframes is explained"),
        (260, 305, "a dense keyframe panel changes into a slot playlist", "hundreds of overlapping keyframes are hard to manage", "What does the overlapping graph panel look like?", "the headache of managing hundreds of keyframes", "Where is the crowded curve editor during the warning about granular control?"),
        (370, 425, "connection-based nodes show loops and several playheads", "multiple simultaneous playheads break linearity", "Show the node graph with looping paths", "multiple playheads at once", "What connected-node picture accompanies the point that timeline linearity no longer translates?"),
        (455, 515, "design-guideline text leads into a familiar three-monitor timeline", "the proposal must remain familiar, intuitive, flexible, and node based", "Which three numbered preview windows are on the editor?", "What guidelines constrain the proposed editor?", "Find the three-preview mockup when familiarity with traditional editors is emphasized"),
        (545, 610, "video and audio nodes feed a timeline node", "inputs are added sequentially and a gap becomes its own node", "video nodes feeding the long horizontal timeline", "How does the timeline node place incoming clips?", "Which graph demonstrates sequential placement while a blank gap is described as a node?"),
        (635, 705, "trim and transition nodes appear between clips", "later clips ripple automatically when trim length changes", "Could you locate the purple transition node between clips?", "automatic ripple after changing a trim", "What transition graphic remains connected as the speaker explains automatic shifting?"),
        (805, 890, "multicam footage and a camera-selection node are shown", "camera angles can change without destroying the underlying edit", "How many camera views are visible in the multicam preview?", "non-destructive multicam reordering", "Take me to the multicam selector while the presenter explains preserving the original edit"),
        (1060, 1145, "nested graphs expose inputs, outputs, and a skin-tone node", "a preset can be injected in the middle and shared across tools", "nested node file with a skin-tone modifier", "How could separate creative applications exchange node files?", "Which nested graph is displayed while cross-application inputs and outputs are proposed?"),
    ],
    "ui-frameworks-talk": [
        (45, 68, "a Coollab screenshot sits beside a red flower photo", "Jules identifies himself as Coollab's developer", "Who is pictured beside the Coollab interface?", "Coollab developer introduction", "Find the red flower and Coollab screen shown as Jules introduces his role"),
        (105, 165, "a dense ImGui panel and a hand-drawn grid layout are contrasted", "complex wrapping and alignment require manual calculation", "the rough sketch with a purple tile in the center", "Why did ImGui layouts become a problem?", "What drawing is on screen when automatic thumbnail wrapping is described?"),
        (185, 245, "a colorful C-layout poster represents Clay", "Clay supplies rectangles and positions on top of other UI systems", "Could you show me the bright C-layout poster?", "Clay as a layout engine", "Which poster appears while the one-hour Clay prototype is discussed?"),
        (315, 390, "HTML, CSS, JavaScript, React, Svelte, and Vue logos line the slide", "the web offers widespread tooling and flexible interfaces", "What six web technology logos are lined up?", "advantages of choosing web technologies", "Find the row of framework logos while broad web tooling is praised"),
        (470, 535, "an Electron logo is added below the web stack", "Electron packages web frontends as desktop applications", "Where does the Electron badge appear on the slide?", "desktop applications made with Electron", "What logo is added when VS Code and Discord are given as desktop examples?"),
        (555, 625, "C++ and web options show Ultralight, Sciter, and CEF", "the Chromium Embedded Framework is lower level than Electron", "the slide with three C++ web framework logos", "Why is CEF less straightforward than Electron?", "Which three logos appear as native C++ embedding is compared with Electron?"),
        (615, 700, "Tauri and Dioxus pages sit under a Rust plus Web heading", "Tauri uses Rust in the backend and web technology in front", "What is shown beneath the Rust plus Web title?", "Tauri's frontend and backend languages", "Show the paired framework pages while the Rust backend architecture is explained"),
        (770, 835, "the same Rust and web comparison remains visible", "duplicating state and synchronizing every change is identified as the main cost", "Can you find the slide saying one codebase, every platform?", "state synchronization between Rust and JavaScript", "Which slogan remains visible during the warning about duplicated types and callbacks?"),
        (1185, 1260, "ImGui code and interface screenshots return", "immediate mode redraws each frame without change callbacks", "the blue ImGui window next to source code", "How does immediate-mode UI avoid change tracking?", "Find the ImGui screenshot when rendering every frame is explained"),
        (1285, 1390, "an egui web page and Rust crab images appear", "egui is chosen for better layouts while keeping immediate-mode simplicity", "Which library name sits above two red crab icons?", "final reason for selecting egui", "Show the egui page while the presenter explains the compromise he accepted"),
    ],
})


def build() -> dict:
    root = Path(__file__).resolve().parent
    source_doc = json.loads((root / "sources_v1.json").read_text(encoding="utf-8"))
    sources = {source["source_id"]: source for source in source_doc["sources"]}
    queries = []
    route_fields = (("VISUAL", 4), ("SPEECH", 5), ("HYBRID", 6))
    for source_id, events in EVENTS.items():
        source = sources[source_id]
        for event_index, event in enumerate(events, 1):
            start, end, visual, speech, *_ = event
            for route, query_index in route_fields:
                query = event[query_index]
                ambiguous = route != "HYBRID" and event_index in {3}
                queries.append({
                    "query_id": f"{source_id}-{event_index:02d}-{route.lower()}",
                    "source_id": source_id,
                    "video_title": source["title"],
                    "split": source["split"],
                    "query": query,
                    "route": route,
                    "evidence_interval": {"start_seconds": start, "end_seconds": end},
                    "visual_evidence_rationale": visual if route != "SPEECH" else f"Not required: {visual}, but it does not establish the requested spoken claim.",
                    "speech_evidence_rationale": speech if route != "VISUAL" else f"Not required: {speech}, but the requested visible state is identifiable without it.",
                    "human_grounded": True,
                    "ambiguous": ambiguous,
                    "annotation_notes": ("Borderline phrasing retained and flagged before evaluation; another modality may help, but the labeled modality remains sufficient."
                                         if ambiguous else
                                         "Reviewed against actual five-second frames and the local Whisper transcript before model evaluation."),
                })
    return {"schema_version": "1.0.0", "dataset_id": "human-grounded-router-v1",
            "annotation_status": "frozen-before-model-evaluation", "queries": queries}


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "annotations_v1.json"
    target.write_text(json.dumps(build(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
