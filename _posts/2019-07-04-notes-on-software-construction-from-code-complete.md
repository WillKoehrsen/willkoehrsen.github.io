---
published: true
title: "Notes On Software Construction From Code Complete"
date: 2019-07-04
categories:
  - software engineering
  - learning
---
![](https://miro.medium.com/max/2000/1*uo10PpFyHv2Jsmj20SnvFA.jpeg?q=20)
*[Source](https://www.pexels.com/photo/mountain-covered-with-snow-1125268/?)*

## Lessons from “_Code Complete: A Practical Handbook of Software Construction”_ with applications to data science

When people ask about the hardest part of my job as a data scientist, they often expect me to say building machine learning models. Given that all of our ML modeling is done in about 3 lines:

```

from sklearn import model

model.fit(training_features, training_targets)

predictions = model.predict(testing_features)

```

I reply that machine learning is one of the easier parts of the job. Rather, the hardest part of being a data scientist in industry is the **software engineering** required to build the infrastructure that goes into running machine learning models continuously in production.

Starting out, (at [Cortex Building Intel](https://get.cortexintel.com/?)) I could write a [good Jupyter Notebook](https://github.com/WillKoehrsen/Data-Analysis?) for a one-time machine learning project, but I had no idea what it meant to “run machine learning in production” let alone how to do it. Half a year in, and having built several ML systems making predictions around the clock to help engineers run buildings more efficiently, I’ve learned it takes a _whole lot of software construction_ and a tiny bit of data science. Moreover, while there are not yet standard practices in data science, there are time-tested best practices for writing software that can help you be more effective as a programmer.

With a relative lack of software engineering skills entering my job, I’ve had to learn quickly. Much of that came from interacting with other software engineers and soaking up their knowledge, but some of it has also come from resources such as textbooks and online tutorials. One of those textbooks is the 900-page masterwork on constructing quality software, [_Code Complete: A Practical Handbook of Software Construction_](https://en.wikipedia.org/wiki/Code_Complete?)by Steve McConnell. In this article, I wanted to outline the high-level points regarding software construction I took away from reading this book. These are as follows:

1.  **Thoroughly plan your project before touching a keyboard**
2.  **Write readable code because it’s read more than it’s written**
3.  **Reduce the complexity of your programs to free mental capacity**
4.  **Test and review every line of code in a program**
5.  **Be an egoless programmer**
6.  **Iterate on your designs and repeatedly measure progress**

<!--more-->

Although the second edition of this book was written in 2004, the proven ideas in software engineering haven’t changed in the intervening years and the book is still highly relevant to software engineers today. You won’t learn any specific details about data science in Python from this book, but you will learn the fundamentals of how to plan, structure, build, test, and review software projects. I’ve tried to keep the ideas here at a fairly high-level and plan to go more into depth on specific points in later articles.

![](https://miro.medium.com/max/2000/0*Swn3tGNDzgB3pkRn.jpg?q=20)
*_Code Complete by Steve McConnell (900 pages for only $35!)_*

While these practices may seem irrelevant to data science as _it’s currently taught_, I think data science is hurt by the current lack of emphasis on software engineering. After all, a machine learning system is a software project. If there are tested, best practices for delivering successful software projects, then we as data scientists should be practicing these methods to move beyond Jupyter Notebooks and start delivering machine learning solutions. The six topics covered here are not as exciting as the newest machine learning algorithm, but they are critical ideas for data scientists in industry.

* * *

1.  **Thoroughly plan your project before touching a keyboard**

Just as you would never begin [building a house](https://lab3.co.nz/2018/10/29/software-house-building-metaphor/?) without a blueprint, you should not start coding without a detailed, written design for your program. If you don’t know the intended outcome of your program, you will waste lots of time writing muddled code that accomplishes no particular objective. The problem I often ran into with regards to data science is that diving into a dataset is quite fun. It’s hard to resist the temptation to get your hands on the keyboard, making graphs, looking for anomalies, and hacking together models. After a few hours, you inevitably end up with a lot of messy code and no clear deliverable, a problem that could have been averted by taking the time to plan out your program.

Every software project you take on — and this includes machine learning models — should start with a [problem definition](https://en.wikipedia.org/wiki/Problem_statement?): a high-level statement of the problem to solve. An example from my work regarding a demand forecasting model is “We want to predict building energy demand accurately 7 days in advance to help building engineers prepare for the next week.” This statement captures the desired outcome and the business value of the project.

After the problem statement comes the [project requirements](https://www.tutorialspoint.com/software_engineering/software_requirements.htm?), a set of objectives —at a lower level than the problem statement— that a solution must meet. These can cover metrics — an error of less than 10% — or the end user experience — graphs must clearly show best estimates [and uncertainty](/how-to-generate-prediction-intervals-with-scikit-learn-and-python-ab3899f992ed?). The requirements will guide the detailed design of your program and allow you to assess if the project is a success.

After the description and requirements comes the [architecture specification](https://en.wikipedia.org/wiki/Software_architecture_description?) where you start planning out the files, classes, and routines (functions) that will make up your program. At this point, you can start getting into the details of your design such as error handling, input/output, user interface, and data flow through the program.

These documents should be reviewed and discussed just as much as the actual code because errors and decisions made in the design phase affect the rest of the project. Here is where you need to understand what is being asked of your program, and how you will approach the problem. Plan out where you foresee difficulties, make time estimates for the project, outline alternative approaches, and assign responsibility during the design phase of a project, Only after everyone has agreed to the description, requirements, and architecture plan should you even think of hitting the keyboard.

The exact steps above may change, but the important idea is that you should never start off a project by writing a bunch of code. Even on small, one-off personal data science projects, I now take the time to plan out — and write down — my overall goal and a set of requirements for my program. It’s a good habit to start; remember that large projects don’t arise out of people hacking away on a keyboard, they are planned out and built a piece at a time, following a detailed blueprint drawn up during the design process.

At my current company, all of our machine learning projects involve a substantial design phase where we have discussions with the business side of the company, our clients, and customer success to make sure we meet the needs of our end users. This process usually ends up creating several dozen pages of documentation that we refer to throughout the rest of the project. While it’s true that requirements will change over the course of the project, it’s crucial to have a checklist of what you need your code to accomplish because otherwise you’ll just be hacking and building something that ultimately will not be useable. Design tends to take up about 30% of the average project time for us, a worthwhile investment. Planning and writing down our design ahead of time means that instead of purposelessly banging together hammers and nails when we start coding, we follow an outline and build a sturdy structure a piece at a time.

* * *

**2\. Write readable code because it’s read more than it’s written**

[Code is going to be read many more times than it will be written](https://devblogs.microsoft.com/oldnewthing/20070406-00/?p=27343&), so never sacrifice read-time convenience — how understandable your code is — for write-time convenience — how quickly you can write the code.

Data science teaches some bad practices around code readability, most notably with variable names. While it might be obvious to you that `X` and `y` stand for features and target because you’ve seen this several hundred times, why not call the variables`features` and `target` to help those less familiar with ML syntax? Reading code should not be an exercise in trying to decipher the cryptic made-up language of whoever wrote it.

Improving code readability means using descriptive names for functions, classes, data objects, and any variable in your program! As an example, never use `i`, `j`, and `k` for loop variables. Instead, use what they actually represent: `row_index`, `column_index` , and `color_channel`. Yes, it takes half a second more to type, but using descriptive variable names will save you, and anyone who reads your code, dozens of hours down the line when debugging or trying to modify the code.

Here’s another example of non-descriptive variable names:

```

value = value * 2.25
value = value * 5.89

```

This code also suffers from the use of [“magic” numbers: undefined constant values](https://en.wikipedia.org/wiki/Magic_number_(programming)?) that describe something (a [conversion](https://www1.oanda.com/currency/converter/?), number of sensor reads, a rate, etc.) only no one can tell _what_ because they just appear as if out of magic. You should _never_ have a magic number in your code because they make programs impossible to understand, are a large cause of errors, and make updating these values difficult as you have to hunt for all the places they are used. Instead, take the below approach:

```

total_widgets_sold = hours * widgets_sold_per_hour
total_revenue_from_widgets = total_widgets_sold * price_per_widget

```

The `widgets_per_hour` and `price_per_widget` should be function parameters, or [named constants](http://wiki.c2.com/?NamedConstants=&) if they are used throughout your code (denoted by all [capitals in Python](https://www.programiz.com/python-programming/variables-constants-literals?)). Now, anyone can understand this code, and it’s much easier to modify by changing parameter values.

Other practices encompassed by the idea of making code readable are that the visual structure of your code should reflect the logical structure (for example when writing [loops](https://www.scaler.com/topics/java/loops-in-java/)), that your comments should show the intent of code rather than just stating what the code does, minimizing the [span and live time of a variable](http://www.rosscode.com/blog/index.php?title=variable_usage_span_and_live_time&more=1&c=1&tb=1&pb=1&), grouping together related statements, avoiding overly complicated if-else statements, and keeping functions as short as possible.

If you ever find yourself thinking “I’ll keep these variable names short to save time” or “I’ll go back and document this code when I’m done writing it”, stop, do your future self a favor, and make the code more readable. As McConnell emphasizes, readable code has a positive effect on all the following:

*   Understandability
*   Modifiability
*   Error Rate
*   Debugging
*   Development time — more readable code improves development time over the course of a project
*   External quality — more readable code creates a better product for the end user as a result of the above factors

I think the practice of writing tangled code in data science is a result of individuals developing code intended to run only a single time. While this might be fine for a personal project, in industry, code readability is much more valuable than how quickly you can put together a model. Much of the first code I wrote on the job is unintelligible even to me because I did not think about the people who would be reading it in the future. As a result, this code is not fit for production and sits languishing in unused branches on GitHub. _Remember, don’t ever prioritize writing speed over reading comprehension._

* * *

**3\. Reduce the complexity of your programs to free mental capacity**

As emphasized throughout Code Complete, the [primary imperative of software construction is to manage complexity](https://www.microsoftpressstore.com/articles/article.aspx?p=2222451&seqNum=2&). What does this mean? It’s about limiting the amount of information you have to hold in your head while programming and reducing arbitrary decisions. Instead of expanding your intellectual ability to write more complicated code, simplify your existing code to a level where you can understand it with your current intellect.

As an example of limiting information to recall, consider the situation where you have 2 functions, one to email users and one to send them a text. If you want to make things really hard on yourself, you do the following:

```

def notify_user_method_one(user, message):
    """Send a text to a user"""
    ...

def notify_user_method_two(message, user):
    """Send an email to a user"""
    ...

```

The problem with this code is that you have to remember which function corresponds to which method and the order of variables. A much better approach, resulting in fewer pieces of information to recall is:

```

def text_user(user, message):
    ...

def email_user(user, message):
   ...

```

Now the function name describes exactly what the function does and the arguments are consistent so you don’t waste mental energy thinking about their order.

The **_concept of consistency_** is crucial for reducing code complexity. The argument for having standards/conventions is you don’t have to make multiple small decisions about things tangentially related to coding such as formatting. Pick a standard and apply it across your entire project. Rather than worrying about what capitalization to use for variable names, apply the same rules to all variables in your project and you don’t have to make a decision. The choice of a standard often matters less than the actual standard itself so don’t get too caught up arguing about whether you should use 2 spaces or 4\. Just pick one, set up your development environment to automatically apply it, and go to work.

(I don’t want to get too into specific technologies, but if you use Python, I have to recommend the `[black](https://black.readthedocs.io/en/stable/?)` [autoformatter](https://black.readthedocs.io/en/stable/?). This tool has completely solved our team’s issues with code formatting and styling. We set it up to auto format our code and never have to worry about the length of lines or whether we should put spaces after commas. I have it set to auto-run on save in [vscode](https://code.visualstudio.com/?)).

Other ways you can [reduce complexity](https://codeburst.io/complexity-in-software-architecture-decompose-for-simplicity-22945b4018bf?) is by, providing consistent interfaces to all your functions and classes ([sklearn](https://scikit-learn.org/?) is a great example of this), using the same error handling method everywhere, avoiding deeply nested loops, adopting conventions when possible, keeping functions short. On the subject of functions, make sure that each function does a single task and does it well! The name of a function should be self-documenting and describe exactly the single action done by the function (like `email_user`). If you find yourself writing a function doc-string with the word “and” describing what the function does, you need to simplify the function. Shorter functions that do only one thing are easier to remember, easier to test, reduce the opportunity for errors, and allow for greater modifiability.

You can’t really make yourself much smarter, but you can make your code much simpler, thereby freeing your mental resources to concentrate on solving tough issues. When explaining technical concepts, the mark of a master is not using complicated jargon, but using simple language that anyone can understand. Likewise, when writing code, an experienced developer’s code may perform a complex task, but it will hide that complexity allowing others to understand and build on it. It can be momentarily satisfying to write tricky code that only you understand, but eventually, you’ll realize that an effective programmer writes the simplest code. Reducing complexity increases code quality and limits the number of decisions you have to make so you can focus on the difficult parts of a program.

* * *

**4\. Test and review every line of code in a program**

Testing is one of the most-poorly-covered areas in data science education yet it’s absolutely crucial for production code. Even code written by professional programmers has [15–50 errors in every 1000 lines of code](https://labs.sogeti.com/how-many-defects-are-too-many/?). Testing is one of several techniques to try and find errors or, at the least, assert your program works as intended. Without testing, we cannot release our machine learning models into production due to the risk of unintended failures. One of the quickest ways to lose customers would be to have mission-critical code fail because it was not thoroughly tested.

A good rule of thumb for testing (this technique is called [structural basis testing](https://oraclefrontovik.com/2016/07/26/structured-basis-testing/?)) is you need one test for every `if, for, while, and, or, elif` if your code. At a minimum, you want to test every statement in a program at least once. Our codebase has testing for every function, from loading data, transforming data, feature engineering, modeling, predicting, storing predictions, generating model explanations, and validating models which together cover every line of code in our codebase.

Testing deserves at least its own article (or [probably book)](https://www.oreilly.com/library/view/python-testing-with/9781680502848/?), but a good place to start is with [Pytest](https://doc.pytest.org/?). Fortunately, these modern libraries make setting up and developing tests much less tedious. Furthermore, you can set up pytest (or other frameworks) to automatically run your testing suite with every commit to GitHub through a [Continuous Integration service like CircleCI](https://circleci.com/?).

In addition to (not as a substitute for) testing, every line of code written for a project should be reviewed by multiple programmers. This can be through formal code inspections or informal code reviews where the purpose is to get multiple eyes on the code to flush out errors, check the code logic, enforce consistent conventions, and improve general code quality through feedback. Code reviews are some of the best opportunities to learn, especially if you are inexperienced. Experience is a great teacher, but it requires a long time to acquire. As a shortcut, you can let others hindsight — the mistakes they’ve made — be your foresight by listening to their constructive criticism.

Also, when you know that others will look at your code in a review, it forces you to write better code (think about the difference in quality between the things you write in private and in public). Moreover, once you start thinking about the tests you need to run over your code, it improves initial code quality (some people [even recommend writing tests before you write the code](https://en.wikipedia.org/wiki/Test-driven_development?)).

On our team, we typically spend almost as long testing and reviewing code as we do writing it in the first place to make sure it does exactly what we want with no side effects. At first, this was incredibly frustrating — my typical response was “I wrote the code and it ran once on my machine so why should I test it” — before I realized all the errors I wasn’t catching in my code because I wasn’t testing it. Testing may be foreign to many data scientists, but it’s a proven and universal method in software engineering because it improves code quality and reduces errors.

If you are working on an individual project, you can still add testing and solicit feedback. However, that sometimes can be hard so an alternative is to start [contributing to open source](https://opensource.guide/how-to-contribute/?). Most libraries, especially the major data-science ones, have strict testing and code review requirements. These may be intimidating at first, but realize that procedures exist for a reason — to ensure that code continues running as intended — and that you can’t get better without trying and failing. I think testing has been overlooked in data science because of the lack of deployed machine learning systems. You don’t need testing to compile a Jupyter Notebook, but you sure need testing when your code is helping to run the [largest office buildings in Manhattan!](https://cortexintel.com/?)

* * *

**5\. Be an egoless programmer**

Before every code review, I take time to tell myself: “you’ve made some mistakes in your code. They are going to be pointed out in this review, but don’t take it personally. Own up to your mistakes and use this experience to learn how to become a better programmer.” I have to do this because I find it very hard to admit when I’m wrong and as with most people, I tend to have an initial negative reaction to criticism. However, over time, I’ve learned failing to admit you are wrong and need to change is one of the greatest blockers to getting better at coding (and at any activity).

My interpretation of being an [egoless programmer](https://en.wikipedia.org/wiki/Egoless_programming?) means accepting your failures as a chance to learn. Don’t take feedback on your code personally, and realize that others are genuinely trying to help in code reviews. Egoless programming also means being willing to let go of your beloved frameworks or standards when they become out of date (in other words, don’t be resistant to change). McConnell makes the point that software engineering is a field where 10 years of experience can be worse than 1 year if the person with more experience has not updated her knowledge since she started. The best tools are constantly changing — especially in data science — and standards can also change over time.

This doesn’t mean [jump ship for the newest technology](https://vicki.substack.com/p/you-dont-need-kafka?) immediately, but it does mean if there is a proven benefit to switching, then don’t be so set in your ways that you refuse to change. Software development is not a deterministic process, it is heuristic — driven by rules of thumb — and you have to be willing to try many different approaches rather than sticking with the exact same method. Sometimes this means abandoning one model when it’s clearly not working — even if you’ve spent dozens of hours on it — and accepting other’s solutions when they are demonstrably better than your own.

To extend the “building a house metaphor” from earlier, construction workers do not use a single tool — the hammer — to build a house. Instead, they have a complete toolbox full of different implements for the varied tasks involved in construction. Likewise, in software engineering, or data science, the person who only knows one method will not get very far. If you are an egoless programmer, you’ll be open to learning from others, respond constructively to feedback, and fill your toolbox with the appropriate techniques. **_You’ll get much farther much quicker by admitting your mistakes than by asserting you can never make them._**

* * *

**6\. Iterate on your designs and repeatedly measure progress**

Software development (and data science) is fundamentally an iterative process. Great solutions do not emerge fully-formed from one individual’s fingers the first time they touch a keyboard. Rather, they are developed over long processes, with many repetitions of earlier stages as the design is refined and features are added. Writing good software requires a willingness to keep working at a problem, making code more readable, more efficient, and less error-prone over time by responding to feedback and thinking deeply about problems (sometimes the best tool is a pencil and paper for writing down your thoughts). Don’t expect to get things completely right the first time!

Iteration — and this is probably a familiar concept to data scientists — should be informed by repeated measurements. Without measuring, you are effectively blind. How do you know whether an optimization increased the speed of your code? How do you know which parts of the code are the most error-prone? How do you know which features users spend the most time with? How do you know which parts of a project take the most time? The answer is that you collect data and analyze it. Based on the results, you then improve your approach on the next iteration.

Throughout _Code Complete_, McConnell stresses the need for measurement to make better decisions. Whenever we see a process that could be made more efficient, we need to look for opportunities where a little data can help us optimize. For example, when I started tracking my time in my first few months on the job, I noticed I was spending more than 75% of my coding time on writing and debugging tests. This was an unacceptably large share, so I decided to spend time reading how to write good unit tests, practicing writing tests, and I started to think about the tests I would write before coding. As a result, I reduced the percent of time writing tests down to less than 50% and was able to spend more time understanding the problem domain (another critical aspect of data science that is hard to teach).

The most important part of tracking data is to look at _relative changes_ over time. In most cases, the absolute value of an observation is not as important as the _change of that value relative to the last time you measured it_. Noticed that your model performance has been decreasing over time? Well, maybe that’s because one of the building’s power meters has gone down and needs to be fixed. Tracking outcomes over time requires only setting up a system that records data and making sure someone is checking it periodically.

Measurements should help inform all aspects of the software construction process from design to code tuning. When estimating how long a project will take, you should look at past estimates and see why they were inaccurate. If you want to try and optimize your code (always make sure your code is working before [trying to improve the performance](https://softwareengineering.stackexchange.com/questions/80084/is-premature-optimization-really-the-root-of-all-evil?)) you have to measure each incremental change. There are numerous examples in the book that point out supposed performance enhancements that actually had the opposite effect! If you don’t measure the effects of a change, you cannot know if what you are doing is really worthwhile.

The goal of data science isn’t to collect data and build nice graphs, it’s to make better decisions and improve processes through data. You can apply this to [your own work](http://acqnotes.com/acqnote/tasks/measures-of-effectivenessrequirements?) by tracking your development habits, figuring out where you are weakest, and focusing on that area for improvement. Track changes over time to make sure you’re headed in the right direction and course correct as often as necessary.

* * *

## Putting These Ideas into Practice

We’ve walked through some of the key ideas on software construction at a high level and the next step is to put them into practice by actually writing (better) code. First, realize that you won’t be able to adopt these all at once: as with any profession, improving at coding takes time. (Peter Norvig has a great essay [on how to learn to program in 10 years](https://norvig.com/21-days.html?), a more realistic goal than “learning Python in 24 hours”.) Instead, focus on one or two ideas at a time, and try to put them into practice either at work or on personal projects.

If you have the opportunity to learn from others at work, then take full advantage of that (assuming they are using best practices) by adopting an egoless attitude. If you are learning on your own, take a look at getting involved with open source. There are plenty of projects looking for help. If that’s a little intimidating, you can try just reading some of the code in well-written libraries. (Some Python examples [listed in the Hitchhiker’s Guide to Python](https://docs.python-guide.org/writing/reading/?) are: [Flask](https://github.com/mitsuhiko/flask?), [Werkzeug](https://github.com/mitsuhiko/werkzeug?), [Requests](https://github.com/kennethreitz/requests?), and [Diamond](https://github.com/python-diamond/Diamond?)).

# Conclusions

The overall theme I took away from the 900+ pages of [_Code Complete_](https://www.microsoftpressstore.com/store/code-complete-9780735619678?)is that quality software is produced through a rigorous design and development process. That rigor is often missing from data science, which tends towards convoluted code to get a solution once, rather than code that can be run millions of times without error. Many people who come into data science — myself included — lack the formal training in computer science and software engineering best practices. However, these programming practices are relatively simple to pick up and will pay off far down the road in terms of your ability to write production-level data science code.

Quality software development is a process, and I’m hoping that data scientists start to adopt thorough processes that allow them to translate their work into deliverable products. Sure it’s exciting when you develop a new AI that can play computer games [better than a human](https://techcrunch.com/2013/04/14/nes-robot/?), but it’s even cooler when your code helps the [Empire State Building to save almost a million dollars a year](https://www.businesswire.com/news/home/20180207005941/en/Empire-State-Building-Cortex-Mobile-App-Reduce?). The field of data science will move [past the hype stage of the curve](https://en.wikipedia.org/wiki/Hype_cycle?) when it proves it can deliver useful products with business value, as software engineering has done for several decades. [Data science can have a massive](https://matrix.berkeley.edu/initiatives/matrix-seminars/data-science-and-climate-change?) impact on the real world but that won’t happen until data scientists use practices that allow our code to withstand the rigors of the real world.

* * *

I write about data science and sometimes [other interesting activities](https://medium.com/@williamkoehrsen/100-miles-through-the-park-what-its-like-to-run-a-100-mile-ultramarathon-1ab36c46d76d?). The best place to follow me is on Twitter [@koehrsen_will](http://twitter.com/@koehrsen_will?).
