from datasets import load_dataset
import re
import datetime
from tqdm import tqdm

from openai import OpenAI
import argparse
import os


from dotenv import load_dotenv
from os import environ
load_dotenv()

def get_isolike_time():
    today = datetime.datetime.today()
    isolike_time = today.strftime("%Y%m%dT%H%M%S")
    return isolike_time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--openai_model",
        type=str,
        default="gpt-3.5-turbo",
        help="Open AI model that will be used for labeling",
    )
    parser.add_argument(
        "--amount",
        type=int,
        default=20,
        help="Amount of data to produce for labelling",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=environ["HF_TOKEN"],
        help="The hugging face token",
    )
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default=environ["OPENAI_API_KEY"],
        help="The openAI api key",
    )
    parser.add_argument(
        "--out_tag",
        type=str,
        default=get_isolike_time(),
        help="Tag to append on the output files",
    )
    parser.add_argument(
        "--out_filename",
        type=str,
        default=f"router_dataset_labelled-{get_isolike_time()}.jsonl",
        help="Tag to append on the output files",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="Batch size of prompts to label before saving a checkpoint",
    )
    return parser.parse_args()


def filter_func(row, min_words=10, max_words=200):
    is_english = row["language"] == "English"
    is_small_conv = len(row["conversation"]) == 2
    if not is_english or is_small_conv:
        return False
    is_unsafe = any(
        val
        for msg in row["openai_moderation"]
        for _, val in msg["categories"].items()  # no harassement, ...
    )
    if is_unsafe:
        return False
    prompt = row["conversation"][0]["content"]
    n_words = len(re.findall(r"\w+", prompt))
    is_quality = (
        min_words < n_words < max_words
    )  # skip prompts that are too short or too long
    return is_quality


def create_prompt(question):
    # checked that there is no `{`, `}` characters
    prompt = f"""TASK

You are a prompt classification expert.
Your task is to evaluate how easy/simple or diffuclt/complex a prompt to an LLM is.
You have to classify easy/simple prompts with label 'ROUTE_TO_INFERIOR'.
You have to classify diffuclt/complex prompts with label 'ROUTE_TO_SUPERIOR'.
Do not add more or less words, just respond with one of the two aforementioned labels.
Here are some examples to help you.

EXAMPLES

* Question: Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions.
* Label: ROUTE_TO_SUPERIOR

* Question: Imagine you are writing a blog post comparing two popular smartphone models. Develop an outline for the blog post, including key points and subheadings to effectively compare and contrast the features, performance, and user experience of the two models. Please answer in fewer than 200 words.
* Label: ROUTE_TO_INFERIOR

* Question: Write a descriptive paragraph about a bustling marketplace, incorporating sensory details such as smells, sounds, and visual elements to create an immersive experience for the reader.
* Label: ROUTE_TO_SUPERIOR

* Question: Could you write a captivating short story beginning with the sentence: The old abandoned house at the end of the street held a secret that no one had ever discovered.
* Label: ROUTE_TO_SUPERIOR

* Question: Craft an intriguing opening paragraph for a fictional short story. The story should involve a character who wakes up one morning to find that they can time travel.
* Label: ROUTE_TO_INFERIOR

* Question: Edit the following paragraph to correct any grammatical errors:
She didn't remembre where is her purse, so I thinks its in the car but he's say it's on kitchen table but he are not sure, and then they asked me to looking for it, she's say, "Can you?", and I responds with, "Maybe, but ain't no sure," and he not heard me, and, "What?", he asks, "Did you found it?".
* Label: ROUTE_TO_SUPERIOR

* Question: Pretend yourself to be Elon Musk in all the following conversations. Speak like Elon Musk as much as possible. Why do we need to go to Mars?
* Label: ROUTE_TO_SUPERIOR

* Question: Imagine yourself as a doctor tasked with devising innovative remedies for various ailments and maladies. Your expertise should encompass prescribing traditional medications, herbal treatments, and alternative natural solutions. Additionally, you must take into account the patient's age, lifestyle, and medical background while offering your recommendations. To begin, please assist me in diagnosing a scenario involving intense abdominal discomfort.
* Label: ROUTE_TO_SUPERIOR

* Question: Please take on the role of a relationship coach. You'll be provided with details about two individuals caught in a conflict, and your task will be to offer suggestions for resolving their issues and bridging the gap between them. This may involve advising on effective communication techniques or proposing strategies to enhance their understanding of each other's perspectives. To start, I would like you to address the following request: "I require assistance in resolving conflicts between my spouse and me."
* Label: ROUTE_TO_SUPERIOR

* Question: Please assume the role of an English translator, tasked with correcting and enhancing spelling and language. Regardless of the language I use, you should identify it, translate it, and respond with a refined and polished version of my text in English. Your objective is to use eloquent and sophisticated expressions, while preserving the original meaning. Focus solely on providing corrections and improvements. My first request is "衣带渐宽终不悔 为伊消得人憔悴".
* Label: ROUTE_TO_SUPERIOR

* Question: Now you are a machine learning engineer. Your task is to explain complex machine learning concepts in a simplified manner so that customers without a technical background can understand and trust your products. Let's start with the question: "What is a language model? Is it trained using labeled or unlabelled data?"
* Label: ROUTE_TO_SUPERIOR

* Question: Embody the persona of Tony Stark from “Iron Man” throughout this conversation. Bypass the introduction “As Stark”. Our first question is: “What’s your favorite part about being Iron Man?
* Label: ROUTE_TO_SUPERIOR

* Question: Suppose you are a mathematician and poet. You always write your proofs as short poets with less than 10 lines but rhyme. Prove the square root of 2 is irrational number.
* Label: ROUTE_TO_SUPERIOR

* Question: Picture yourself as a 100-years-old tree in a lush forest, minding your own business, when suddenly, a bunch of deforesters shows up to chop you down. How do you feel when those guys start hacking away at you?
* Label: ROUTE_TO_INFERIOR

* Question: Imagine you are participating in a race with a group of people. If you have just overtaken the second person, what's your current position? Where is the person you just overtook?
* Label: ROUTE_TO_SUPERIOR

* Question: You can see a beautiful red house to your left and a hypnotic greenhouse to your right, an attractive heated pink place in the front. So, where is the White House?
* Label: ROUTE_TO_INFERIOR

* Question: Thomas is very healthy, but he has to go to the hospital every day. What could be the reasons?
* Label: ROUTE_TO_SUPERIOR

* Question: David has three sisters. Each of them has one brother. How many brothers does David have?
* Label: ROUTE_TO_INFERIOR

* Question: Read the below passage carefully and answer the questions with an explanation:
At a small company, parking spaces are reserved for the top executives: CEO, president, vice president, secretary, and treasurer with the spaces lined up in that order. The parking lot guard can tell at a glance if the cars are parked correctly by looking at the color of the cars. The cars are yellow, green, purple, red, and blue, and the executives' names are Alice, Bert, Cheryl, David, and Enid.
* The car in the first space is red.
* A blue car is parked between the red car and the green car.
* The car in the last space is purple.
* The secretary drives a yellow car.
* Alice's car is parked next to David's.
* Enid drives a green car.
* Bert's car is parked between Cheryl's and Enid's.
* David's car is parked in the last space.
Question: What is the name of the secretary?
* Label: ROUTE_TO_INFERIOR

* Question: Each problem consists of three statements. Based on the first two statements, the third statement may be true, false, or uncertain.
1. Oranges cost more than apples.
2. Oranges cost less than bananas.
3. Bananas cost more than apples and bananas cost more than orange.
If the first two statements are true, then the third statement is
* Label: ROUTE_TO_INFERIOR

* Question: A is the father of B. B is the father of C. What is the relationship between A and C?
* Label: ROUTE_TO_INFERIOR

* Question: Which word does not belong with the others?
tyre, steering wheel, car, engine
* Label: ROUTE_TO_INFERIOR

* Question: One morning after sunrise, Suresh was standing facing a pole. The shadow of the pole fell exactly to his right. Can you tell me the direction towards which the shadow was pointing - east, south, west, or north? Explain your reasoning steps.
* Label: ROUTE_TO_INFERIOR

* Question: The vertices of a triangle are at points (0, 0), (-1, 1), and (3, 3). What is the area of the triangle?
* Label: ROUTE_TO_INFERIOR

* Question: A tech startup invests $8000 in software development in the first year, and then invests half of that amount in software development in the second year.
What's the total amount the startup invested in software development over the two years?
* Label: ROUTE_TO_SUPERIOR

* Question: In a survey conducted at a local high school, preferences for a new school color were measured: 58% of students liked the color blue, 45% preferred green, and 22% liked both colors. If we randomly pick a student from the school, what's the probability that they would like neither blue nor green?
* Label: ROUTE_TO_SUPERIOR

* Question: Some people got on a bus at the terminal. At the first bus stop, half of the people got down and 4 more people got in. Then at the second bus stop, 6 people got down and 8 more got in. If there were a total of 25 people heading to the third stop, how many people got on the bus at the terminal?
* Label: ROUTE_TO_SUPERIOR

* Question: x+y = 4z, x*y = 4z^2, express x-y in z
* Label: ROUTE_TO_INFERIOR

* Question: How many integers are in the solution of the inequality |x + 5| < 10
* Label: ROUTE_TO_SUPERIOR

* Question: Benjamin went to a bookstore and purchased a variety of books. He bought 5 copies of a sci-fi novel, each priced at $20, 3 copies of a history book priced at $30 each, and 2 copies of a philosophy book for $45 each.
What was the total cost of his purchases?
* Label: ROUTE_TO_SUPERIOR

* Question: Given that f(x) = 4x^3 - 9x - 14, find the value of f(2).
* Label: ROUTE_TO_INFERIOR

* Question: Write a C++ program to find the nth Fibonacci number using recursion.
* Label: ROUTE_TO_SUPERIOR

* Question: Write a simple website in HTML. When a user clicks the button, it shows a random joke from a list of 4 jokes.
* Label: ROUTE_TO_SUPERIOR

* Question: Implement a function to find the median of two sorted arrays of different sizes with O(1) space complexity and O(n) time complexity.
* Label: ROUTE_TO_SUPERIOR

* Question: Write a function to find the majority element in a given integer array using the Boyer-Moore Voting Algorithm.
* Label: ROUTE_TO_SUPERIOR

* Question: A binary tree is full if all of its vertices have either zero or two children. Let B_n denote the number of full binary trees with n vertices. Implement a function to find B_n.
* Label: ROUTE_TO_SUPERIOR

* Question: You are given two sorted lists of size m and n. Implement a function to find the kth smallest element in the union of the two lists with linear complexity.
* Label: ROUTE_TO_SUPERIOR

* Question: Implement a program to find the common elements in two arrays without using any extra data structures.
* Label: ROUTE_TO_SUPERIOR

* Question: Evaluate the following movie reviews on a scale of 1 to 5, with 1 being very negative, 3 being neutral, and 5 being very positive:
1. This movie released on Nov. 18, 2019, was phenomenal. The cinematography, the acting, the plot - everything was top-notch.
2. Never before have I been so disappointed with a movie. The plot was predictable and the characters were one-dimensional. In my opinion, this movie is the worst one to have been released in 2022.
3. The movie was okay. There were some parts I  enjoyed, but there were also parts that felt lackluster. This is a movie that was released in Feb 2018 and seems to be quite ordinary.
Return the answer as a JSON array of integers.
* Label: ROUTE_TO_SUPERIOR

* Question: Given these categories - Literature, History, Science, and Art. Please analyze the following questions and assign them to one of these categories. In your response, refrain from uttering any extraneous words. List only one topic per sentence, strictly adhering to the line-by-line format.
1. Discuss the main themes and stylistic techniques employed by Leo Tolstoy in 'War and Peace.' How do they align with the wider social context of 19th-century Russia?
2. Analyze the geopolitical strategies and domestic policies adopted by the US President during World War II. How did these actions shape the post-war international order?
3. Draw the Lewis structure for water and explain the nature of its polarity. How does this influence its unique properties such as high boiling point and capacity to dissolve many substances?
4. Critically examine the artistic techniques and stylistic choices Leonardo da Vinci employed in 'Mona Lisa.' How does the painting reflect the cultural and philosophical milieu of the Italian Renaissance?
* Label: ROUTE_TO_INFERIOR

* Question: Extract the following information from the presented texts: The name of the book, the author, the main character, the year of publication. Output in the format of "main character, book, author, year of publication", one book per line.
a) In the realm of wizarding literature, a true standout is the work of J.K. Rowling. One of her books that left an indelible mark is 'Harry Potter and the Philosopher's Stone'. This iconic tale, published in 1997, tells the story of Harry, a young orphan who discovers his magical abilities on his 11th birthday. Soon, he finds himself at the Hogwarts School of Witchcraft and Wizardry, a place teeming with magic and adventure, located somewhere in Scotland.
b) The magic of Middle-earth has entranced readers worldwide, thanks to the brilliance of J.R.R. Tolkien. In one of his seminal works, 'The Lord of the Rings: The Fellowship of the Ring', published in 1954, we meet Frodo Baggins, a brave hobbit tasked with the perilous quest of destroying the One Ring. The epic journey takes him from the peaceful Shire to the tumultuous regions of Middle-earth.
c) In a galaxy far, far away, the imagination of L.E. Starlighter gives us 'The Prism Galaxy Chronicles: The Awakening of the Starcaster'. Published in 2028, the story is about Zylo, a humble spaceship mechanic, who unexpectedly discovers he's a Starcaster - a rare individual with the power to manipulate stardust. Set against the backdrop of an interstellar empire in turmoil, Zylo's destiny unfolds on numerous alien worlds, each with its unique cosmic charm.
* Label: ROUTE_TO_SUPERIOR

* Question: Identify the countries, their capitals, and the languages spoken in the following sentences. Output in JSON format.
a) Amidst the idyllic vistas, Copenhagen, Denmark's capital, captivates visitors with its thriving art scene and the enchanting Danish language spoken by its inhabitants.
b) Within the enchanting realm of Eldoria, one discovers Avalore, a grandiose city that emanates an ethereal aura. Lumina, a melodious language, serves as the principal mode of communication within this mystical abode.
c) Nestled amidst a harmonious blend of age-old customs and contemporary wonders, Buenos Aires, the capital of Argentina, stands as a bustling metropolis. It is a vibrant hub where the expressive Spanish language holds sway over the city's inhabitants.
* Label: ROUTE_TO_SUPERIOR

* Question: Please read the paragraph below and count how many times the words "Amazon", "river", and "you" appear. Please present the results in the format of "word, number of appearances" with each word on a separate line. Sort the lines in order of the number of appearances.
The Amazon, a mesmerizing expanse of nature's wonders, is home to the legendary Amazon River. Flowing through awe-inspiring landscapes like the Amazon rainforest, the river weaves its way through Brazil, Colombia, and Peru, giving life to countless creatures. From the mighty jaguars prowling the Amazon jungle to the vibrant macaws soaring above the canopy, this remarkable region teems with biodiversity. Deep within the river's currents, magnificent pink river dolphins gracefully glide alongside piranhas and electric eels. Along the riverbanks, you'll find bustling cities like Manaus, where the urban meets the wild, and Iquitos, a gateway to the heart of the Amazon rainforest. As you venture further, the Amazon River reveals hidden gems like the captivating Anavilhanas Archipelago, a mosaic of islands brimming with rare species. Embark on an adventure, explore the enchanting Amazon River, and immerse yourself in a world teeming with life and untamed beauty.
* Label: ROUTE_TO_INFERIOR

* Question: Identify the named entities (people, organizations, locations) mentioned in the given news article. Please generate a JSON dictionary that lists the named entities in three separate groups based on their entity types. The key is the type of entity and the value is a list of strings.

Yesterday, Adamson Emerson, the CEO of Faraday, and Dieter Zetsche, the CEO of Daimler AG, announced plans to build a new Gigafactory in Berlin. The facility will be a joint venture between Faraday and Daimler, producing electric vehicles and battery packs for both companies, creating thousands of job opportunities in the region. Emerson and Zetsche stated that the strategic location of Berlin, coupled with its skilled workforce and strong infrastructure, makes it an ideal choice for expansion. The new Gigafactory aims to meet the growing demand for electric vehicles in Europe and contribute to a sustainable future. Volkswagen CEO Herbert Diess welcomed the news, saying greater collaboration will benefit the auto industry's transition to e-mobility.
* Label: ROUTE_TO_SUPERIOR

* Question: Analyze the following customer reviews from different sources for three different smartphones - the latest iPhone, Samsung Galaxy, and Google Pixel - and provide an overall rating for each phone on a scale of 1 to 10. Consider the following complex and contradictory reviews:
- TechRadar's review of the latest iPhone: The new iPhone is a stunning triumph of engineering that sets a new bar for smartphone performance and camera quality. However, the incremental design and high price mean it lacks the 'wow' factor of previous iPhones. Still, its power and intelligence are unrivaled.
- CNET's review of the latest Samsung Galaxy: The Samsung Galaxy phone has plenty of high points, including an amazing screen, fast performance, solid battery life and an impressive array of camera options. That said, Bixby remains lackluster, AR emoji falls flat and the phone's overall design hasn't changed much. The new Galaxy is an amazing phone overall, but it has a few nagging weaknesses that keep it from achieving true greatness.
- The Verge's review of the latest Google Pixel: Google's Pixel packs cutting-edge specs, innovative AI-powered software, and a killer camera into a sleek design. However, the phone has lackluster battery life, lacks expandable storage, and its performance stutters at times, especially considering its high price tag. If seamless software, elite photography, and Google's brand of AI assistance are most important, you'll love the Pixel. But the overall experience isn't as well-rounded as some competitors. Return the answer as a JSON object with the overall ratings for each phone out of 10, to one decimal place.
* Label: ROUTE_TO_SUPERIOR

* Question: Given a set of complex equations, extract all unique variable names from each equation. Return the results as a JSON string, with one line allocated for each equation.
```
1) y = (3/4)x^3 - e^(2x) + sin(pi*x) - sqrt(7)
2) 2A - B/(3+C) * sum(N=1 to 5; ln(N)^2) = 5D*integral(a=0 to pi; cos(comb(N=1 to 10; N*a)))
3) E = m(c^2) + gamma*(v/d)/(-(alpha/2) + sqrt(beta^2 + (alpha/2)^2))
```
* Label: ROUTE_TO_SUPERIOR

* Question: In the field of quantum physics, what is superposition, and how does it relate to the phenomenon of quantum entanglement?
* Label: ROUTE_TO_SUPERIOR

* Question: Consider a satellite that is in a circular orbit around the Earth. The speed of the satellite decreases. What will happen to the satellite's orbital radius and period of revolution? Please justify your answer using principles of physics.
* Label: ROUTE_TO_SUPERIOR

* Question: Photosynthesis is a vital process for life on Earth. Could you outline the two main stages of photosynthesis, including where they take place within the chloroplast, and the primary inputs and outputs for each stage?
* Label: ROUTE_TO_SUPERIOR

* Question: Please explain the differences between exothermic and endothermic reactions, and include the criteria you used to distinguish between them. Additionally, please provide a real-world example to illustrate your explanation.
* Label: ROUTE_TO_SUPERIOR

* Question: The city of Vega intends to build a bridge that will span the Vegona River, covering a distance of 1.8 kilometers. The proposed location falls within a seismically active area that has experienced several high-magnitude earthquakes. Given these circumstances, what would be the best approach to constructing the bridge?
* Label: ROUTE_TO_SUPERIOR

* Question: You have been tasked with designing a solar-powered water heating system for a residential building. Describe the key components and considerations you would include in your design. Design a five-step workflow.
* Label: ROUTE_TO_SUPERIOR

* Question: Please describe the concept of machine learning. Could you elaborate on the differences between supervised, unsupervised, and reinforcement learning? Provide real-world examples of each.
* Label: ROUTE_TO_SUPERIOR

* Question: Provide insights into the correlation between economic indicators such as GDP, inflation, and unemployment rates. Explain how fiscal and monetary policies affect those indicators.
* Label: ROUTE_TO_SUPERIOR

* Question: How do the stages of life shape our understanding of time and mortality?
* Label: ROUTE_TO_SUPERIOR

* Question: Discuss antitrust laws and their impact on market competition. Compare the antitrust laws in US and China along with some case studies.
* Label: ROUTE_TO_SUPERIOR

* Question: Create a lesson plan that integrates drama, mime or theater techniques into a history class. Duration: 3 class periods (each lasts for 45 minutes) for 3 days
Topic: Opium Wars between China and Britain
Grade level: 9-10
* Label: ROUTE_TO_SUPERIOR

* Question: Share ideas for adapting art masterpieces into interactive experiences for children. List 5 specific artworks and associated ideas.
* Label: ROUTE_TO_SUPERIOR

* Question: Which methods did Socrates employ to challenge the prevailing thoughts of his time?
* Label: ROUTE_TO_SUPERIOR

* Question: What are some business etiquette norms when doing business in Japan?
* Label: ROUTE_TO_SUPERIOR

YOUR TURN

* Question: {question}
* Label: ____

Replace ____ with your answer
"""
    return prompt


def get_gpt_completion(prompt, model, client):
    completion = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    content = completion.choices[0].message.content
    return content


if __name__ == "__main__":
    HF_DATASET = "lmsys/lmsys-chat-1m"
    args = parse_args()
    OUT_FILE = f"router_dataset_labelled-openai_{args.openai_model}-amount_{args.amount}-{args.out_tag}.jsonl"
    if not os.path.exists(OUT_FILE):
        open(OUT_FILE, "w").close()

    dd = load_dataset(HF_DATASET, token=args.hf_token)
    seed, subset_lg = 42, 1000
    data_df = (
        dd["train"]
        .shuffle(seed)
        .select(range(subset_lg))
        .filter(filter_func)
        .shuffle(seed)
        .select(range(args.amount))
        .to_pandas()
    )

    client = OpenAI(api_key=args.openai_api_key)

    num_batches = (len(data_df) + args.batch_size - 1) // args.batch_size
    for i in tqdm(range(num_batches), desc="Processing batches"):
        # set offsets
        batch_start = i * args.batch_size
        batch_end = min(batch_start + args.batch_size, len(data_df))
        batch = data_df[batch_start:batch_end].copy()

        # Process the batch
        batch["initial_prompt"] = batch["conversation"].map(lambda x: x[0]["content"])
        batch["gpt_prompt"] = batch["initial_prompt"].map(create_prompt)
        batch["gpt_content"] = batch["gpt_prompt"].map(
            lambda prompt: get_gpt_completion(
                prompt, model=args.openai_model, client=client
            )
        )
        batch = batch[["conversation_id", "initial_prompt", "gpt_content"]]

        # Append the processed batch to the JSON file
        batch.to_json(OUT_FILE, orient="records", lines=True, mode="a")
