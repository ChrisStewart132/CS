public class Singleton {
    /*
     * Subclassing Singleton is possible (unlike all-static approach), but does require more elaborate
     * initialization of uniqueInstance. Life’s too short, use this format to transform any class to a Singleton
     */
    private static Singleton uniqueInstance;

    Singleton() {
        // Private constructor to prevent instantiation
        // Singleton singleton = Singleton.getInstance();// to create / get ref to the Singleton
    }

    public static synchronized Singleton getInstance() {
        // Watch for thread safety — getInstance() should be synchronised; uniqueInstance volatile
        
        if (uniqueInstance == null) {
            uniqueInstance = new Singleton();// can create a Singleton privately
        }
        return uniqueInstance;
    }

    // additional functions below..
}
