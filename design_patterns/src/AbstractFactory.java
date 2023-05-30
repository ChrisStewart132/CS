/* Factory Method AKA Virtual Constructor

 * Problem:
    • Normally, code that expects an object of a particular class does not need to know
        which subclass the object belongs to.
    • E.g. a Player in an adventure game uses a Weapon, and does not need to know exactly what
        kind of Weapon it is.
    • Exception: when you create an object, you need to know its exact class. The
        “new” operator increases coupling!
    • Need a way to create the right kind of object, without knowing its exact class


    Solution:
        • Move the “new” into an abstract method. (Can be parameterized.)
        • Override that method to create the right subclass object.


    Notes:
        • It is common to have more than one factory method.
            E.g. weaponFactory(), treasureFactory(), potionFactory()
        • Swing UIManager.getUI(Jcomponent) is a fancy factory method.
        • Java 11 API has 89 classes with “Factory” in their names


    GOF/Abstract        Concrete
    --------------------------------
    AbstractCreator     Player
    ConcreteCreator     Warrior
    ConcreteCreator     Wizard
    AbstractProduct     Weapon
    ConcreteProduct     Sword
    ConcreteProduct     Wand
    FactoryMethod       makeWeapon()

 */

interface AbstractProduct {

}

interface AbstractCreator {
    void makeWeapon();
    void attack();
}

interface IAbstractFactory {
    AbstractProduct createProductA();
    AbstractProduct createProductB();
}

public class AbstractFactory implements IAbstractFactory {

    @Override
    public AbstractProduct createProductA() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'createProductA'");
    }

    @Override
    public AbstractProduct createProductB() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'createProductB'");
    }
    
}
