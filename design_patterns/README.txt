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