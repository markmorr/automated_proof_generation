# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 13:41:01 2022

@author: 16028
"""

import os
import openai



ENGINE = "ada"
openai.api_key = "MY_API_KEY_DONT_PUSH_TO_GITHUB"
openai.api_key = "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"
openai.Engine.list()
openai.Completion.create(
    model="ada:ft-columbia-university-2022-04-03-22-12-33",
    prompt="Tell me what year it is.")
openai.Engine(ENGINE).search(    
  documents=["White House", "hospital", "school"],
  query="the president"
)



class Example():
    """Stores and input, output pair and formats it to prime the model."""
    def __init__(self, inp, out):
        self.input = inp
        self.output = out 
        
    def get_input(self):
        return self.input
    
    def get_output(self):
        return self.output
    
    def format(self):
        return f"input: {self.input}\noutput: {self.output}\n"
    

ex1 = Example("dog", "to the cat")
ex1.format()

#citing Caz Czworkowski on writing this class to use the GPT AI, https://www.youtube.com/watch?v=_N1dEamEqpg
class GPT:
    def __init__(self, engine=ENGINE,
                 temperature=.5,
                 max_tokens=100,
                 frequency_penalty= .5):
    
    

        self.examples = []
        self.engine = engine
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.frequency_penalty = frequency_penalty
        
    def add_example(self, ex):
        self.examples.append(ex)
        
    def get_prime_text(self, ex):
        return '\n'.join(self.examples) + '\n'
    
    def run_search(self, ex):
        response = openai.Completion.create(engine=self.get_engine(),
                                            promppt = self.craft_query(prompt),
                                            max_tokens = self.get_max_tokens,
                                            temperature = self.get_temperature,
                                            top_p = 1,
                                            frequency_penalty = self.get_frequency_penalty,
                                            n=1,
                                            stream=False,
                                            stop = "\ninput:")
        
    
 
prompt1 = {"prompt": "Prove that the mean is an unbiased estimator for the Gaussian distribution", 
 "completion": "Using the Central Limit Theorem, we see that the sample mean tends towards the true mean with probability 1."}

classifier = GPT()
classifier.add_example(prompt1)


classifier = GPT.add_example(prompt1)


myproblemlist = [
{
    "problem": "A 4-inch by 6-inch picture is enlarged for framing  by tripling its dimensions.  A 2-inch-wide border  is then placed around each side of the enlarged  picture, as shown.  Thin metal framing is sold only  in increments of one foot.  What is the minimum  number of linear feet of framing that must be  purchased to go around the perimeter of the border?\n\n[asy]\n\ndraw((0,0)--(14,0)--(14,20)--(0,20)--cycle,linewidth(2));\n\ndraw((4,4)--(10,4)--(10,16)--(4,16)--cycle);\n\nlabel(\"border\",(7,17),N);\n\nlabel(\"picture\",(7,8),N);\n\nlabel(\"frame\",(14,5),E);\n\ndraw((17.5,7.5)--(14.5,7.5),Arrow);\ndraw((10.5,7.5)--(13.5,7.5),Arrow);\n\n[/asy]",
    "level": "Level 5",
    "type": "Prealgebra",
    "solution": "After the picture is enlarged by tripling its dimensions, the dimensions become $12\\times18$. After the border is added, the dimensions of the picture increase to $16\\times22$ (since each side has a 2-inch border). The perimeter is $16+16+22+22=76$ inches. Since $76/12=6\\frac{1}{3}$, we need $\\boxed{7}$ feet of framing to go around the entire picture."
},



{
    "problem": "What is the sum of the reciprocals of the natural-number factors of 6?",
    "level": "Level 2",
    "type": "Number Theory",
    "solution": "The natural-number factors of 6 are 1, 6, 2, 3. The sum of their reciprocals is $1/1+1/6+1/2+1/3=6/6+1/6+3/6+2/6=12/6=\\boxed{2}$."
},


{
    "problem": "Given that $a$ is a multiple of $1428$, find the greatest common divisor of $a^2+9a+24$ and $a+4$.",
    "level": "Level 4",
    "type": "Number Theory",
    "solution": "We can use the Euclidean Algorithm. \\begin{align*}\n&\\text{gcd}\\,(a^2+9a+24,a+4) \\\\\n&\\qquad=\\text{gcd}\\,(a^2+9a+24-(a+5)(a+4),a+4)\\\\\n&\\qquad=\\text{gcd}\\,(a^2+9a+24-(a^2+9a+20),a+4)\\\\\n&\\qquad=\\text{gcd}\\,(4,a+4).\n\\end{align*} Since $4$ is a factor of $a$ and thus $a+4$, the greatest common divisor is $\\boxed{4}$."
}  , 


{
    "problem": "What is the remainder when 5462 is divided by 9?",
    "level": "Level 1",
    "type": "Number Theory",
    "solution": "A number is congruent to the sum of its own digits modulo 9.  (In other words, if you have a number $n$, and the sum of its digits is $m$, then $n$ and $m$ leave the same remainder when divided by 9.)\n\nThe sum of the digits of 5462 is $5 + 4 + 6 + 2 = 17$, and the sum of the digits of 17 is $1 + 7 = 8$.  Therefore, the remainder when 5462 is divided by 9 is $\\boxed{8}$."
},


{
    "problem": "Find the least positive integer $x$ that satisfies $x+4609 \\equiv 2104 \\pmod{12}$.",
    "level": "Level 4",
    "type": "Number Theory",
    "solution": "Subtract 4609 from both sides of the congruence to obtain $x\\equiv -2505\\pmod{12}$. By dividing 2505 by 12, we find that the least integer $k$ for which $-2505+12k>0$ is $k=209$. Adding $12\\cdot 209$ to $-2505$, we find that $x\\equiv 3\\pmod{12}$. Thus $\\boxed{3}$ is the least integer satisfying the given congruence."
},


{
    "problem": "How many different prime factors does $20!$ have? (Reminder: If $n$ is a positive integer, then $n!$ stands for the product $1\\cdot 2\\cdot 3\\cdot \\cdots \\cdot (n-1)\\cdot n$.)",
    "level": "Level 3",
    "type": "Number Theory",
    "solution": "$20!=20\\cdot19\\cdot18\\cdot...\\cdot3\\cdot2\\cdot1$ is divisible by every prime less than 20. There are $\\boxed{8}$ such primes: 2, 3, 5, 7, 11, 13, 17, 19."
},

{
    "problem": "If $k$ and $\\ell$ are positive 4-digit integers such that $\\gcd(k,\\ell)=3$, what is the smallest possible value for $\\mathop{\\text{lcm}}[k,\\ell]$?",
    "level": "Level 5",
    "type": "Number Theory",
    "solution": "The identity $\\gcd(k,\\ell)\\cdot\\mathop{\\text{lcm}}[k,\\ell] = k\\ell$ holds for all positive integers $k$ and $\\ell$. Thus, we have $$\\mathop{\\text{lcm}}[k,\\ell] = \\frac{k\\ell}{3}.$$Also, $k$ and $\\ell$ must be 4-digit multiples of $3$, so our choices for each are $$1002,1005,1008,1011,1014,\\ldots,$$and by minimizing the product $k\\ell$, we minimize the least common multiple of $k$ and $\\ell$. However, $k$ and $\\ell$ cannot both be $1002$, since their greatest common divisor would then be $1002$ (not $3$). Setting $k=1002$ and $\\ell=1005$, we obtain $\\gcd(k,\\ell)=3$ as desired, and we obtain the smallest possible value for the least common multiple: \\begin{align*}\n\\mathop{\\text{lcm}}[1002,1005] &= \\frac{1002\\cdot 1005}{3} \\\\\n&= 1002\\cdot 335 \\\\\n&= (1000\\cdot 335)+(2\\cdot 335)\\\\\n&= \\boxed{335{,}670}.\n\\end{align*}"
},

{
    "problem": "A school has between 150 and 200 students enrolled. Every afternoon, all the students come together to participate in gym class. The students are separated into six distinct sections of students. If one student is absent from school, the sections can all have the same number of students. What is the sum of all possible numbers of students enrolled at the school?",
    "level": "Level 4",
    "type": "Number Theory",
    "solution": "If there are $s$ students, then $s-1$ must be divisible by 6. In other words, we want to find the sum of all values of $s$ for which $s-1\\equiv 0\\pmod{6}$. The multiples of 6 in the given range are 150, 156, ..., 198, so the possible values of $s$ are 151, 157, ..., 199. Recalling that the sum of an arithmetic series is  \\[\n\\frac{(\\text{first term}+\\text{last term})(\\text{number of terms})}{2},\n\\]we find that these integers sum to $(151+199)(9)/2=\\boxed{1575}$."
},


{
    "problem": "The numbers 60, 221, and 229 are the legs and hypotenuse of a right triangle.  Find the multiplicative inverse to 450 modulo 3599.  (Express your answer as an integer $n$ with $0\\leq n<3599$.)",
    "level": "Level 4",
    "type": "Number Theory",
    "solution": "We notice that $450=221+229,$ so that must be the connection.  The Pythagorean Theorem tells us  \\[60^2+221^2=229^2\\] so \\[229^2-221^2=60^2.\\] The difference of squares factorization tells us  \\[(229-221)(229+221)=3600\\] and taken modulo 3599 we get  \\[8\\cdot450\\equiv1\\pmod{3599}.\\] The answer is $\\boxed{8}$."
}
]


curated_list = []
for problem in myproblemlist:
    curr_dict = dict()
    curr_dict['prompt'] = problem['problem']
    curr_dict['completion'] = problem['solution']
    curated_list.append(curr_dict)
    
    
import glob
directoryPath = r'C:\Users\16028\Downloads\MATH\MATH\train\intermediate_algebra'


for filename in res:
    i+= 1
    print(i)
    x = pd.read_csv(filename)
    df_dict[i] = x
    glued_data = pd.concat([glued_data,x],axis=0)
    
  
import pandas as pd
import json
dfs = [] # an empty list to store the data frames
file_list = [f for f in glob.glob(directoryPath+'\*.json')]# or "123" in f or "a1b" in f]

for file in file_list:
    with open(file) as f:
        json_data = pd.json_normalize(json.loads(f.read()))
    # data = pd.read_json(file, lines=True) # read data frame from json file
    dfs.append(json_data) # append the data frame to the list

temp = pd.concat(dfs, ignore_index=True) # concatenate all the data frames in the list.
    

df = temp[['problem', 'solution']]
df.rename(columns={'problem': 'prompt', 'solution':'completion'})

classif

