// Observer interface
interface Observer {
    void update();
}

// Subject interface
interface Subject {
    void registerObserver(Observer observer);
    void unregisterObserver(Observer observer);
    void notifyObservers();
}