// Observer interface
interface Observer {
    void update(Subject subject, Object data);
}

// Subject interface
interface Subject {
    void registerObserver(Observer observer);
    void unregisterObserver(Observer observer);
    void notifyObservers();
    void setData(Object data);
}


