// code
package com.example.mypackage;
    Package names should be unique to avoid naming conflicts with other packages.

    The package declaration should match the directory structure of the source file. For example, a file in the package com.example.mypackage should be located
    in a directory structure like com/example/mypackage.

    Classes in the same package can refer to each other without explicit import statements. However, to use classes from other packages, you need to import them using
    the import statement.

    Packages can be nested, allowing for further categorization and organization of classes.

    Java provides a standard set of packages, such as java.lang, java.util, and java.io, which contain commonly used classes and interfaces.

// compile
javac <source_file>.java
javac <source_file1>.java <source_file2>.java ...
javac -d <output_directory> <source_file1>.java <source_file2>.java ...

// run
java <main_class>

// create executable (jar)
jar cf YourJarName.jar package1 package2 ...
jar cf YourJarName.jar File1.class File2.class
jar cfe YourJarName.jar com.example.Main Main.class OtherClass.class


// design patterns
The Gang of Four (GoF) design patterns are categorized into three main types: creational patterns, structural patterns, and behavioral patterns. Here's a brief overview of each category:
    Creational Patterns:
        Singleton: 
		Ensure a class only has one instance, and provide a global point of access to it.

        Factory Method: 
		Define an interface for creating an object, but let subclasses decide which class to instantiate. Factory method lets a class defer instantiation to subclasses.

        Abstract Factory: 
		Provide an interface for creating families of related or dependent objects without specifying their concrete classes.

        Builder: 
		Separate the construction of a complex object from its representation so that the same construction process can create different representations.

        Prototype: 
		Creates new objects by cloning existing ones, avoiding the need for explicit class instantiation.

    Structural Patterns:
        Adapter: 
		Converts the interface of a class into another interface that clients expect, enabling classes with incompatible interfaces to work together.

        Bridge: 
		Separates an abstraction from its implementation, allowing them to vary independently.

        Composite: 
		Composes objects into tree structures to represent part-whole hierarchies, letting clients treat individual objects and compositions uniformly.

        Decorator: 
		Dynamically adds responsibilities to objects by wrapping them in an object of a decorator class.

        Facade: 
		Provides a unified interface to a set of interfaces in a subsystem, simplifying complex systems by providing a higher-level interface.

        Flyweight: 
		Shares common state among multiple objects to reduce memory usage when many similar objects are needed.

        Proxy: 
		Provides a surrogate or placeholder for another object to control access to it.

    Behavioral Patterns:
        Chain of Responsibility:
		Allows an object to pass a request along a chain of potential handlers until one of them handles the request.

        Command: 
		Encapsulates a request as an object, thereby letting you parameterize clients with different requests, queue or log requests, and support undoable operations.

        Interpreter: 
		Defines a representation of a grammar or language and provides an interpreter to evaluate or interpret sentences in the language.

        Iterator: 
		Provides a way to access the elements of an aggregate object sequentially without exposing its underlying representation.

        Mediator: 
		Defines an object that encapsulates how a set of objects interact, promoting loose coupling by keeping objects from referring to each other explicitly.

        Memento: 
		Captures and restores an object's internal state, without violating encapsulation, so that the object can be restored to its previous state.

        Observer: 
		Defines a one-to-many dependency between objects, so that when one object changes state, all its dependents are notified and updated automatically.

        State: 
		Allows an object to alter its behavior when its internal state changes, encapsulating the possible behavior within separate state classes.

        Strategy: 
		Defines a family of algorithms, encapsulates each one, and makes them interchangeable, allowing algorithms to vary independently from clients that use them.

        Template Method: 
		Defines the skeleton of an algorithm in a method, deferring some steps to subclasses, allowing subclasses to redefine certain steps of the algorithm without changing its structure.

        Visitor: 
		Separates an algorithm from the objects it operates on, allowing the same algorithm to work with different object structures.