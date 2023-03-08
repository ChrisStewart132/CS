"""
AGILE
    Agile Software Development is a philosophy.
        Four core values
            • individuals and interactions over processes and tools
            • working software over comprehensive documentation
            • customer collaboration over contract negotiation
            • responding to change over following a plan

    Twelve Agile principles
         1: “Our highest priority is to satisfy the customer through early and continuous delivery of
            valuable software.”        
         2: “Welcome changing requirements, even late in development. Agile processes harness change for
            the customer’s competitive advantage.”        
         3: “Deliver working software frequently, from a couple of weeks to a couple of months, with a
            preference to the shorter timescale.”        
         4: “Business people and developers must work together daily throughout the project.”       
         5: “Build projects around motivated individuals. Give them the environment and support they need,
            and trust them to get the job done.”        
         6: “The most efficient and effective method of conveying information to and within a development
            team is face-to-face conversation.”        
         7: “Working software is the primary measure of progress.”        
         8: “Agile processes promote sustainable development. The sponsors, developers, and users should
            be able to maintain a constant pace indefinitely.”       
         9: “Continuous attention to technical excellence and good design enhances agility.”        
        10: “Simplicity – the art of maximizing the amount of work not done – is essential.”        
        11: “The best architectures, requirements, and designs emerge from self-organizing teams.”        
        12: “At regular intervals, the team reflects on how to become more effective, then tunes and
            adjusts its behavior accordingly.”      
            
SCRUM
    Scrum is a fast product development method
        • initiated by Takeuchi and Nonaka Takeuchi and Nonaka, “The New New Product Development Game”, 1986
        • team commitment and empowerment
        • knowledge goes up in the hierarchy
        • a reader’s digest and reflection by Jason Yip (Spotify)
    Adapted to software development
        • objectives achieved within sprint deliveries
        • a sprint has a goal
        • improved code quality and documentation
        • quick intro on scrum website
        • Beedle et al., “SCRUM: An extension pattern language for hyperproductive software development”, 1999
    Values
        • Openness
        • Focus
        • Respect
        • Courage
        • Commitment
    Cycle
        • Sprint planning
        • Sprint execution
        • Sprint review
        • Sprint retrospective
    Roles
        Product owner (PO) a.k.a. customer voice
            • translates user requirements into user stories
            • maintains and prioritises the list of things to do (a.k.a. backlog)
            • negotiates content of releases and timing with the team
        Scrum master (SM) a.k.a. process leader and facilitator
            • acts as a coach, i.e. does not command
            • facilitates communication inside and outside the team
            • represents management, but protects team members
        Team a.k.a. developers
            • everyone is a developer (often even the SM), no hierarchy
            • self-organising and cross-functional (i.e. low bus factor)
            • collective responsibility for achievements (i.e. snow ploughing)

Lean
    Optimisation of processes (more than agile)
        • seven types of waste (based on Toyota 1940’s manufacturing)
        → transport, inventory, motion, waiting, over processing, over production, defects
        • a quick intro at roadmunk.com
        • Poppendieck and Poppendieck, Lean Software Development: An Agile Toolkit, 2003
    Seven principles
        1: optimise the whole avoid thinking on partial optimisations
        2: focus on customer understand and answer to customer needs
        3: energise workers unhappy or unrewarded team mates aren’t performing
        4: eliminate waste avoid over-engineering
        5: lean first welcome changing requirements
        6: deliver fast time-to-market is critical
        7: keep getting better focus on people delivering results instead of the results

Kanban
    Upfront planning is reduced to its minimal
        • continuous changes in one global board
        • quick intro at kanbanize.com
        • Anderson, Kanban: Successful Evolutionary Change for Your Technology Business, 2010
    Kanban proposes interesting metrics
        WIP:
            work in progress is limited to avoid too many things in the pipeline
        queue:
            measures efficiency by comparing tasks in WIP and waiting in queue
        throughput:
            average number of work units processed per unit of time. e.g., tasks per day, story points per weeks...
        lead time:
            total time between a customer demand arrives and is deployed
        cycle time:
            processing time for a task in (a set of) states, removing queuing time
            → often use variation of Little’s law on queuing : CycleTime = WIP / Throughput



It all starts from a vision
    • creates an initial backlog with user stories and acceptance criteria
    • requires some grooming (continuously -Kanban-, before a sprint -Scrum-)
        • Scrum larger initial effort is needed before the development starts
        • Kanban task-oriented, starts as soon as we have work to do
    • priorities must be clearly stated by the PO

Sprint Planning - two parts
    Part I - what are we going to do?
        • the PO presents the highest priority stories
        • the team estimates the complexity of each story
            - planning poker
        • use previous sprints as reference (i.e. we’ll talk about velocity)
        → a sprint should be as coherent as possible (i.e. sprint goal)
    Part II - how are we going to do it?
        • selected stories are broken down into tasks by the team
        • tasks are described thouroughly, i.e. everyone can pick it up
        • estimations of tasks are set collaboratively but not the whole team at once
         
        
Sprint Execution
    • Sprint board (product backlog, sprint backlog, doing, review, done)
        • tasks for stories in the product backlog that pass the definition of ready (DoR) can be moved to doing
        
Sprint Review finished tasks or sprints
    At the end of a task (Kanban) or sprint (Scrum)
        • demonstrate the outcomes to the product owner (and stakeholders)
        • gather feedback from them and keep trace of it
    Be kind, be interesting, be prepared, be focused
        • follow a clear scenario by combining user stories logically
        • use realistic test data, but don’t flood with data
        • ensure you have what you need for the PO to get a hand on your system
        • explain the impediments and how you tackled them
        • possibly include a coming soon... section

Sprint Retrospection
    Starfish method:
        more of, less of, keep doing, stop doing, and start doing
    Bubble method:
        join in pairs and retrospect actionable solutions, pairs combine until entire team together

    Values:
        openess:
            welcome comments and constructive critics
        courage:
            is needed by all team members to face issues
        respect:
            always be constructive and respectful in your comments
        focus:
            come prepared with SMART action points
        commitment:
            commit yourselves to resolve any issues
    Issues:
        processes:
            method, standards, communication,...
        scope:
            is the product vision still clear to everyone?
        quality:
            is the product hitting the expected level of quality?
        environment:
            is the team dynamic becoming toxic?
        skills:
            is training or external expertise needed?

            
"""
